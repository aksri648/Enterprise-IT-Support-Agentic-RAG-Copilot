import logging
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()

_active_model: str | None = None
_active_llm: ChatOpenAI | None = None
_web_search = None

def is_model_not_found_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in [
        "404", "model_not_found", "does not exist", "not have access", "decommissioned", "deprecated"
    ])

def get_candidate_models() -> list[str]:
    api_key = (settings.openai_api_key or settings.groq_api_key).strip()
    base_url = (settings.openai_base_url or "").strip()
    is_groq = bool(settings.groq_api_key) or api_key.startswith("gsk_") or "groq.com" in base_url

    candidates = []
    if is_groq:
        # Check user-configured model first (avoiding decommissioned llama-3.3-70b-versatile)
        configured = settings.groq_model or settings.openai_model
        if configured and configured != "llama-3.3-70b-versatile":
            candidates.append(configured)
        # Supported production models on Groq
        for m in ["openai/gpt-oss-120b", "llama-3.1-8b-instant", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]:
            if m not in candidates:
                candidates.append(m)
        # Fallback to llama-3.3-70b-versatile only as last option (for enterprise/whitelisted keys)
        if "llama-3.3-70b-versatile" not in candidates:
            candidates.append("llama-3.3-70b-versatile")
    else:
        # OpenAI or OpenAI-compatible endpoint
        configured = settings.openai_model
        # If user inadvertently left a llama model when using OpenAI endpoint, discard it
        if configured and not configured.lower().startswith("llama-"):
            candidates.append(configured)
        for m in ["gpt-4o-mini", "gpt-4o"]:
            if m not in candidates:
                candidates.append(m)
    return candidates

def _create_llm_instance(model_name: str) -> ChatOpenAI:
    api_key = (settings.openai_api_key or settings.groq_api_key).strip()
    if not api_key:
        raise RuntimeError("API key is missing: please set OPENAI_API_KEY or GROQ_API_KEY")

    base_url = (settings.openai_base_url or "").strip()
    is_groq = bool(settings.groq_api_key) or api_key.startswith("gsk_") or "groq.com" in base_url
    if not base_url and is_groq:
        base_url = "https://api.groq.com/openai/v1"

    kwargs = {
        "model": model_name,
        "temperature": 0,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)

def llm() -> ChatOpenAI:
    global _active_llm, _active_model
    if _active_llm is None:
        candidates = get_candidate_models()
        _active_model = candidates[0]
        _active_llm = _create_llm_instance(_active_model)
    return _active_llm

def call_llm_with_fallback(invoke_fn):
    global _active_llm, _active_model
    candidates = get_candidate_models()

    start_idx = 0
    if _active_model in candidates:
        start_idx = candidates.index(_active_model)

    last_exc = None
    for idx in range(start_idx, len(candidates)):
        model_name = candidates[idx]
        try:
            if _active_llm is None or _active_model != model_name:
                _active_model = model_name
                _active_llm = _create_llm_instance(model_name)
            return invoke_fn(_active_llm)
        except Exception as exc:
            if is_model_not_found_error(exc) and idx + 1 < len(candidates):
                next_model = candidates[idx + 1]
                logger.warning(
                    f"Model '{model_name}' failed with model-not-found error ({exc}). "
                    f"Automatically switching to '{next_model}'..."
                )
                last_exc = exc
                _active_model = None
                _active_llm = None
                continue
            raise exc

    if last_exc:
        raise last_exc

def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search

def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]

def route_question(state: AgentState):
    route = "kb"
    try:
        def _invoke(model_inst):
            router = model_inst.with_structured_output(RouteDecision, method="json_mode")
            return router.invoke(f"""
You route messages for an enterprise IT support assistant.
Use kb for questions about company IT policies, VPN, password reset, MFA, laptop setup,
software access, security, email, devices, troubleshooting, or technology support.
Use direct only for greetings, thanks, or casual chat that needs no company knowledge.
Question: {state['question']}
Return valid JSON like {{"route":"kb"}}.
""")
        decision = call_llm_with_fallback(_invoke)
        if hasattr(decision, "route"):
            route = decision.route
        elif isinstance(decision, dict) and "route" in decision:
            route = decision["route"]
    except Exception as exc:
        logger.warning(f"Structured route fallback triggered: {exc}")
        route = "kb"
    return {"source_used": route, "trace": add_trace(state, f"Router → {route.upper()}")}

def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"

def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(state["current_query"])
    return {"kb_docs": docs, "trace": add_trace(state, f"Private KB retrieval → {len(docs)} chunks")}

def grade_kb(state: AgentState):
    grade_val = "weak"
    try:
        context = "\n\n".join(f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in state["kb_docs"])
        def _invoke(model_inst):
            grader = model_inst.with_structured_output(EvidenceGrade, method="json_mode")
            return grader.invoke(f"""
You grade evidence for an enterprise IT support assistant.
Question: {state['question']}
Private company KB evidence:\n{context}
Return good only if the evidence is sufficient to answer confidently and specifically.
Otherwise return weak. JSON: {{"grade":"good"}} or {{"grade":"weak"}}.
""")
        decision = call_llm_with_fallback(_invoke)
        if hasattr(decision, "grade"):
            grade_val = decision.grade
        elif isinstance(decision, dict) and "grade" in decision:
            grade_val = decision["grade"]
    except Exception as exc:
        logger.warning(f"Structured grade_kb fallback triggered: {exc}")
        grade_val = "good" if state.get("kb_docs") else "weak"
    return {"kb_grade": grade_val, "trace": add_trace(state, f"KB evidence grade → {grade_val.upper()}")}

def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"

def search_web(state: AgentState):
    result = web_search_tool().invoke({"query": state["current_query"]})
    lines, citations = [], []
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + result["answer"])
        for item in result.get("results", []):
            title, url, content = item.get("title", ""), item.get("url", ""), item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            citations.append({"title": title or url, "url": url, "type": "web"})
    else:
        lines.append(str(result))
    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, "Web fallback → Tavily search"),
    }

def grade_web(state: AgentState):
    grade_val = "weak"
    try:
        def _invoke(model_inst):
            grader = model_inst.with_structured_output(EvidenceGrade, method="json_mode")
            return grader.invoke(f"""
Question: {state['question']}
Web evidence:\n{state['web_results']}
Return good if the evidence is sufficient and directly relevant; otherwise weak.
Return valid JSON like {{"grade":"good"}}.
""")
        decision = call_llm_with_fallback(_invoke)
        if hasattr(decision, "grade"):
            grade_val = decision.grade
        elif isinstance(decision, dict) and "grade" in decision:
            grade_val = decision["grade"]
    except Exception as exc:
        logger.warning(f"Structured grade_web fallback triggered: {exc}")
        grade_val = "good" if state.get("web_results") else "weak"
    return {"web_grade": grade_val, "trace": add_trace(state, f"Web evidence grade → {grade_val.upper()}")}

def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"

def rewrite_query(state: AgentState):
    prompt = f"""
Rewrite this IT support question for better private knowledge retrieval and vendor web search.
Preserve intent, add useful technical keywords, do not answer, return only the query.
Question: {state['question']}
"""
    rewritten = call_llm_with_fallback(lambda m: m.invoke(prompt)).content.strip()
    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite → {rewritten}"),
    }

def generate_from_kb(state: AgentState):
    context = "\n\n".join(f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in state["kb_docs"])
    prompt = f"""
You are an enterprise IT support copilot. Answer ONLY from the private company KB below.
Be concise, actionable, and safe. If steps are present, present them clearly.
Do not invent policy details. Mention that the answer is based on the company's private knowledge base.
Question: {state['question']}\n\nPrivate KB:\n{context}
"""
    answer = call_llm_with_fallback(lambda m: m.invoke(prompt)).content
    citations = []
    seen = set()
    for d in state["kb_docs"]:
        src = d.metadata.get("source", "Private KB")
        if src not in seen:
            seen.add(src)
            citations.append({"title": src.split("/")[-1], "url": "", "type": "private_kb"})
    return {"answer": answer, "source_used": "private_kb", "citations": citations, "trace": add_trace(state, "Answer generation → PRIVATE KB")}

def generate_from_web(state: AgentState):
    prompt = f"""
You are an enterprise IT support copilot. The private company KB was insufficient.
Answer ONLY from the web evidence below. Clearly say this is external web information and may need IT validation before changing company-managed systems.
Question: {state['question']}\n\nWeb evidence:\n{state['web_results']}
"""
    answer = call_llm_with_fallback(lambda m: m.invoke(prompt)).content
    return {"answer": answer, "source_used": "web_search", "trace": add_trace(state, "Answer generation → WEB SEARCH")}

def direct_answer(state: AgentState):
    prompt = f"Respond briefly and naturally to: {state['question']}"
    answer = call_llm_with_fallback(lambda m: m.invoke(prompt)).content
    return {"answer": answer, "source_used": "direct", "trace": add_trace(state, "Direct response → no retrieval")}

def insufficient(state: AgentState):
    return {
        "answer": "I couldn't find enough reliable evidence in the company knowledge base or external search to answer confidently. Please contact the IT help desk or provide more details.",
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
    }

def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb", "search_web": "search_web"
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web", "rewrite_query": "rewrite_query", "insufficient": "insufficient"
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile()

agent_graph = build_graph()

def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial)
