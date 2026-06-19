from __future__ import annotations

import json
from typing import Any, TypedDict

from retrieval import format_sources, retrieve
from tools import (
    extract_ticker,
    stock_quote,
    get_company_info,
    get_financial_metrics,
    get_stock_history,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    kiwoom_get_stock_info,
    kiwoom_get_daily_chart,
    _kiwoom_configured,
    _kr_code_from_symbol,
    web_search,
)
import bot_core


class AgentState(TypedDict, total=False):
    message: str
    session_id: str
    image_b64: str | None
    image_mime: str | None
    contexts: list[dict[str, Any]]
    tool_results: list[str]
    answer: str
    ticker: str | None


# ---------------- Single-agent (legacy) flow ----------------

def tool_node(state: AgentState) -> AgentState:
    tool_results = bot_core.maybe_run_tools(state["message"])
    state["tool_results"] = tool_results
    return state


def retrieval_node(state: AgentState) -> AgentState:
    state["contexts"] = retrieve(
        state["message"],
        strategy=bot_core.CONFIG.get("retrieval_strategy", "hybrid_rerank"),
        k=int(bot_core.CONFIG.get("top_k", 5)),
        embedding_model=bot_core.CONFIG.get("embedding_model", "text-embedding-3-small"),
    )
    return state


def answer_node(state: AgentState) -> AgentState:
    prompt = bot_core.build_user_prompt(
        state["message"],
        state.get("contexts", []),
        state.get("tool_results", []),
        image_attached=bool(state.get("image_b64")),
        session_id=state.get("session_id", "default"),
    )
    state["answer"] = bot_core.call_gpt(prompt, image_b64=state.get("image_b64"), image_mime=state.get("image_mime"))
    bot_core.remember(state.get("session_id", "default"), "user", state["message"])
    bot_core.remember(state.get("session_id", "default"), "assistant", state["answer"])
    return state


def build_graph():
    from langgraph.graph import END, StateGraph
    graph = StateGraph(AgentState)
    graph.add_node("tools", tool_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("tools")
    graph.add_edge("tools", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


# ---------------- Multi-agent (research-grade) flow ----------------
# Sequence: data_collect → financial_collect → news_collect → retrieve → synthesize
# Each sub-agent collects a specific subset of tool data, mirroring the
# sample/financial-analyst layout (data_analyst / financial_analyst / news_analyst).


def data_collect_node(state: AgentState) -> AgentState:
    """Sub-agent: 회사 정보 + 시세 + 추세 + 재무지표."""
    msg = state.get("message", "")
    ticker = extract_ticker(msg)
    state["ticker"] = ticker
    results: list[str] = list(state.get("tool_results", []))
    if ticker:
        kr_code = _kr_code_from_symbol(ticker)
        if kr_code and _kiwoom_configured():
            # 국내 종목 → 키움 REST API
            try: results.append("[kiwoom_stock_info] " + json.dumps(kiwoom_get_stock_info(ticker), ensure_ascii=False))
            except Exception as e: results.append(f"[kiwoom_stock_info_error] {e}")
            try: results.append("[kiwoom_daily_chart_30d] " + json.dumps(kiwoom_get_daily_chart(ticker, 30), ensure_ascii=False))
            except Exception as e: results.append(f"[kiwoom_daily_chart_error] {e}")
            # 보조: yfinance 회사 개요 (영문 종목명 + 업종)
            try: results.append("[company_info] " + json.dumps(get_company_info(ticker), ensure_ascii=False))
            except Exception as e: results.append(f"[company_info_error] {e}")
        else:
            # 해외 종목 → yfinance
            try: results.append("[company_info] " + json.dumps(get_company_info(ticker), ensure_ascii=False))
            except Exception as e: results.append(f"[company_info_error] {e}")
            try: results.append("[stock_quote] " + json.dumps(stock_quote(ticker), ensure_ascii=False))
            except Exception as e: results.append(f"[stock_quote_error] {e}")
            try: results.append("[stock_history_1y] " + json.dumps(get_stock_history(ticker, "1y"), ensure_ascii=False))
            except Exception as e: results.append(f"[stock_history_error] {e}")
            try: results.append("[financial_metrics] " + json.dumps(get_financial_metrics(ticker), ensure_ascii=False))
            except Exception as e: results.append(f"[financial_metrics_error] {e}")
    state["tool_results"] = results
    return state


def financial_collect_node(state: AgentState) -> AgentState:
    """Sub-agent: 손익계산서 + 재무상태표 + 현금흐름."""
    ticker = state.get("ticker")
    results: list[str] = list(state.get("tool_results", []))
    if ticker:
        try: results.append("[income_statement] " + json.dumps(get_income_statement(ticker), ensure_ascii=False))
        except Exception as e: results.append(f"[income_statement_error] {e}")
        try: results.append("[balance_sheet] " + json.dumps(get_balance_sheet(ticker), ensure_ascii=False))
        except Exception as e: results.append(f"[balance_sheet_error] {e}")
        try: results.append("[cash_flow] " + json.dumps(get_cash_flow(ticker), ensure_ascii=False))
        except Exception as e: results.append(f"[cash_flow_error] {e}")
    state["tool_results"] = results
    return state


def news_collect_node(state: AgentState) -> AgentState:
    """Sub-agent: 뉴스 / 웹 검색."""
    msg = state.get("message", "")
    ticker = state.get("ticker") or ""
    query = f"{ticker} stock news" if ticker else msg
    results: list[str] = list(state.get("tool_results", []))
    try:
        results.append("[web_search] " + json.dumps(web_search(query, max_results=5), ensure_ascii=False))
    except Exception as e:
        results.append(f"[web_search_error] {e}")
    # Calculator / calorie tools still get a chance to fire on text intent.
    extra = bot_core.maybe_run_tools(msg)
    # avoid duplicating tools already collected by data/financial/news sub-agents
    blocked = ("[stock_quote]", "[company_info]", "[financial_metrics]", "[income_statement]",
               "[balance_sheet]", "[cash_flow]", "[stock_history", "[web_search]")
    for t in extra:
        if not any(t.startswith(b) for b in blocked):
            results.append(t)
    state["tool_results"] = results
    return state


def synthesize_node(state: AgentState) -> AgentState:
    """Final answer composition — reuses answer_node which builds the full prompt + GPT call."""
    return answer_node(state)


def build_multi_agent_graph():
    from langgraph.graph import END, StateGraph
    graph = StateGraph(AgentState)
    graph.add_node("data_collect", data_collect_node)
    graph.add_node("financial_collect", financial_collect_node)
    graph.add_node("news_collect", news_collect_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("synthesize", synthesize_node)
    graph.set_entry_point("data_collect")
    graph.add_edge("data_collect", "financial_collect")
    graph.add_edge("financial_collect", "news_collect")
    graph.add_edge("news_collect", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def invoke_graph(message: str, session_id: str = "default", image_b64: str | None = None, image_mime: str | None = None) -> dict[str, Any]:
    use_multi = bool(bot_core.CONFIG.get("multi_agent_mode")) and bool(extract_ticker(message))
    app = build_multi_agent_graph() if use_multi else build_graph()
    state = app.invoke({"message": message, "session_id": session_id, "image_b64": image_b64, "image_mime": image_mime})
    answer = state.get("answer", "")
    contexts = state.get("contexts", [])
    tool_results = state.get("tool_results", [])
    # GPT는 이미 시스템 프롬프트에 따라 출력 템플릿 형식으로 답변을 만들었음.
    # render_output_template을 한 번 더 적용하면 raw JSON이 답변에 또 끼어들어 가독성이 떨어진다.
    # 그래서 formatted_answer = answer 그대로 사용한다.
    payload = {
        "answer": answer,
        "formatted_answer": answer,
        "sources": format_sources(contexts),
        "tool_results": tool_results,
        "agent_mode": "multi" if use_multi else "single",
    }
    if bot_core.CONFIG.get("show_debug_context"):
        payload["debug_contexts"] = contexts
    return payload
