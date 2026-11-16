import time
from pinecone import Pinecone , ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import get_settings

settings = get_settings()


_embeddings = None
_vectorstore = None


EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-minilm-l6-v2": 384,
    "multilingual-e5-large": 1024,
    "llama-text-embed-v2": 1024,
    "bge-small-en-v1.5": 384,
    "bge-base-en-v1.5": 768,
    "bge-large-en-v1.5": 1024,
}

def get_effective_provider() -> str:
    provider = (settings.embedding_provider or "auto").lower().strip()
    if provider == "auto":
        api_key = (settings.openai_api_key or "").strip()
        base_url = (settings.openai_base_url or "").strip()
        if api_key.startswith("gsk_") or "groq.com" in base_url or not api_key:
            if settings.pinecone_api_key:
                return "pinecone"
            return "fastembed"
        return "openai"
    return provider

def get_effective_model() -> str:
    provider = get_effective_provider()
    model = (settings.embedding_model or "").strip()
    if provider == "pinecone":
        if not model or "text-embedding" in model.lower() or "openai" in model.lower():
            return "multilingual-e5-large"
    elif provider in {"fastembed", "huggingface"}:
        if not model or "text-embedding" in model.lower():
            return "BAAI/bge-small-en-v1.5"
    return model or "text-embedding-3-small"

def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or get_effective_model()).strip()
    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()
    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]
    if "text-embedding-3-small" in normalized or "text-embedding-ada-002" in normalized:
        return 1536
    if "text-embedding-3-large" in normalized:
        return 3072
    if "multilingual-e5" in normalized or "llama-text-embed" in normalized:
        return 1024
    if "all-minilm" in normalized or "bge-small" in normalized:
        return 384
    if "bge-base" in normalized:
        return 768
    if "bge-large" in normalized:
        return 1024

    raise ValueError(
        f"Unsupported embedding model '{name}' for Pinecone. "
        "Add the matching dimension to EMBEDDING_DIMENSIONS."
    )

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        provider = get_effective_provider()
        model = get_effective_model()

        if provider == "pinecone":
            if not settings.pinecone_api_key:
                raise RuntimeError("PINECONE_API_KEY is missing for Pinecone embeddings")
            from langchain_pinecone import PineconeEmbeddings
            _embeddings = PineconeEmbeddings(
                model=model,
                pinecone_api_key=settings.pinecone_api_key,
            )
        elif provider in {"fastembed", "huggingface"}:
            try:
                from langchain_community.embeddings import FastEmbedEmbeddings
                _embeddings = FastEmbedEmbeddings(model_name=model)
            except Exception:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                _embeddings = HuggingFaceEmbeddings(model_name=model)
        elif provider == "openai":
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is missing for OpenAI embeddings")
            kwargs = {
                "model": model,
                "api_key": settings.openai_api_key,
            }
            base_url = settings.embedding_base_url or settings.openai_base_url
            if base_url:
                kwargs["base_url"] = base_url
            _embeddings = OpenAIEmbeddings(**kwargs)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    return _embeddings




def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing_indexes = pc.list_indexes()
    names = [getattr(x, "name", x.get("name") if isinstance(x, dict) else str(x)) for x in existing_indexes]

    if settings.pinecone_index_name in names:
        index_info = pc.describe_index(settings.pinecone_index_name)
        current_dimension = getattr(index_info, "dimension", None)
        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        if current_dimension is not None and current_dimension != desired_dimension:
            pc.delete_index(name=settings.pinecone_index_name)
            time.sleep(1)

    existing_after = pc.list_indexes()
    names_after = [getattr(x, "name", x.get("name") if isinstance(x, dict) else str(x)) for x in existing_after]
    if settings.pinecone_index_name not in names_after:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            time.sleep(1)

    return pc.Index(settings.pinecone_index_name)





def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )
    return _vectorstore



def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.top_k})



def add_documents(chunks):
    store = get_vectorstore()
    return store.add_documents(chunks)
