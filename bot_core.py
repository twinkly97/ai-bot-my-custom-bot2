from __future__ import annotations

import json
import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


from retrieval import format_sources, retrieve
from tools import (
    calorie_tool_notice,
    extract_ticker,
    safe_calculate,
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

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config" / "bot_config.json").read_text(encoding="utf-8"))
_MEMORY: dict[str, deque] = defaultdict(lambda: deque(maxlen=int(CONFIG.get("memory_turns", 6)) * 2 or 1))


def remember(session_id: str, role: str, content: str) -> None:
    if CONFIG.get("use_memory"):
        _MEMORY[session_id].append({"role": role, "content": content})


def memory_text(session_id: str) -> str:
    if not CONFIG.get("use_memory"):
        return ""
    turns = list(_MEMORY.get(session_id, []))
    return "\n".join([f"{t['role']}: {t['content']}" for t in turns[-int(CONFIG.get('memory_turns', 6))*2:]])


def maybe_run_tools(message: str) -> list[str]:
    results = []
    msg_low = message.lower()
    ticker = extract_ticker(message) if CONFIG.get("enable_stock_tool") or CONFIG.get("enable_company_info") or CONFIG.get("enable_financials") else None

    if CONFIG.get("enable_stock_tool") and ticker and any(w in msg_low for w in ["주가", "가격", "price", "stock", "quote", "티커", ticker.lower()]):
        # 한국 종목이면 키움 REST API 우선 (KIWOOM_APP_KEY 셋팅 시), 아니면 yfinance
        kr_code = _kr_code_from_symbol(ticker)
        if kr_code and _kiwoom_configured():
            results.append("[kiwoom_stock_info] " + json.dumps(kiwoom_get_stock_info(ticker), ensure_ascii=False))
            if any(w in msg_low for w in ["추세", "기간", "주간", "월간", "history", "차트", "trend"]):
                results.append("[kiwoom_daily_chart(30d)] " + json.dumps(kiwoom_get_daily_chart(ticker, 30), ensure_ascii=False))
        else:
            results.append("[stock_quote] " + json.dumps(stock_quote(ticker), ensure_ascii=False))
            if any(w in msg_low for w in ["추세", "기간", "주간", "월간", "history", "차트", "trend"]):
                results.append("[stock_history(1mo)] " + json.dumps(get_stock_history(ticker, "1mo"), ensure_ascii=False))

    if CONFIG.get("enable_company_info") and ticker and any(w in msg_low for w in ["회사", "사업", "업종", "섹터", "어떤 회사", "company", "industry", "sector", "business"]):
        results.append("[company_info] " + json.dumps(get_company_info(ticker), ensure_ascii=False))

    if CONFIG.get("enable_company_info") and ticker and any(w in msg_low for w in ["per", "pbr", "배당", "시총", "시가총액", "베타", "valuation", "metrics", "지표"]):
        results.append("[financial_metrics] " + json.dumps(get_financial_metrics(ticker), ensure_ascii=False))

    if CONFIG.get("enable_financials") and ticker:
        if any(w in msg_low for w in ["손익", "income statement", "매출", "영업이익", "순이익", "eps"]):
            results.append("[income_statement] " + json.dumps(get_income_statement(ticker), ensure_ascii=False))
        if any(w in msg_low for w in ["재무상태", "balance sheet", "자산", "부채", "자본"]):
            results.append("[balance_sheet] " + json.dumps(get_balance_sheet(ticker), ensure_ascii=False))
        if any(w in msg_low for w in ["현금흐름", "cash flow", "fcf", "free cash flow", "capex"]):
            results.append("[cash_flow] " + json.dumps(get_cash_flow(ticker), ensure_ascii=False))

    if CONFIG.get("enable_news_search") and any(w in msg_low for w in ["뉴스", "news", "최근", "이슈", "search", "찾아"]):
        query = message
        if ticker:
            query = f"{ticker} stock news"
        results.append("[web_search] " + json.dumps(web_search(query, max_results=5), ensure_ascii=False))

    if CONFIG.get("enable_calculator_tool"):
        calc_match = re.search(r"계산[:：]?\s*([0-9+\-*/(). %]+)", message)
        if calc_match:
            try:
                results.append(f"[calculator] {calc_match.group(1)} = {safe_calculate(calc_match.group(1))}")
            except Exception as exc:
                results.append(f"[calculator_error] {exc}")
    if CONFIG.get("enable_calorie_tool"):
        if any(w in message for w in ["칼로리", "음식", "식단", "calorie", "kcal"]):
            results.append("[calorie_tool] " + calorie_tool_notice())
    return results


def build_user_prompt(message: str, contexts: list[dict[str, Any]], tool_results: list[str], image_attached: bool, session_id: str = 'default') -> str:
    source_text = format_sources(contexts)
    graph_summary = ""
    graph_path = ROOT / "data" / "graph" / "graph_summary.md"
    if CONFIG.get("enable_graphrag") and graph_path.exists():
        graph_summary = graph_path.read_text(encoding="utf-8")[:2500]
    return f"""
사용자 질문:
{message}

대화 메모리:
{memory_text(session_id)}

도구 실행 결과:
{chr(10).join(tool_results) if tool_results else '도구 실행 없음'}

RAG 검색 근거:
{source_text}

GraphRAG 요약:
{graph_summary or 'GraphRAG 컨텍스트 없음'}

이미지 첨부 여부: {image_attached}

출력 템플릿:
{CONFIG.get('output_template', '')}
""".strip()


def safe_format(template: str, **kwargs: Any) -> str:
    """Fill {placeholders} in template, leaving unknown placeholders as '' and tolerating stray braces.

    Plain str.format raises on missing keys or unbalanced braces. This helper keeps demo output
    readable when the user wrote a custom output_template that doesn't match our default keys.
    """
    if not template:
        return ""

    class _SafeDict(dict):
        def __missing__(self, key):  # type: ignore[override]
            return ""

    try:
        return template.format_map(_SafeDict(**kwargs))
    except Exception:
        # Fall back to manual replacement so a broken template never crashes the chat.
        out = template
        for k, v in kwargs.items():
            out = out.replace("{" + k + "}", str(v))
        return out


def call_gpt(prompt: str, image_b64: str | None = None, image_mime: str | None = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "OPENAI_API_KEY가 설정되지 않았습니다. 배포 환경변수 또는 .env에 키를 넣으면 GPT 답변이 생성됩니다.\n\n"
            "아래는 현재 조립된 프롬프트 미리보기입니다.\n\n" + prompt[:3000]
        )
    try:
        from openai import OpenAI
    except Exception:
        return "openai 패키지가 설치되어 있지 않습니다. pip install -r requirements.txt 후 다시 실행하세요.\n\n" + prompt[:3000]
    client = OpenAI(api_key=api_key)
    content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
    if image_b64 and CONFIG.get("enable_vision"):
        mime = image_mime or "image/png"
        content.append({"type": "input_image", "image_url": f"data:{mime};base64,{image_b64}"})
    primary_model = os.getenv("OPENAI_MODEL", CONFIG.get("model_name", "gpt-4o-mini"))
    fallback_models = ["gpt-4o-mini", "gpt-4.1-mini"]
    last_err: Exception | None = None
    for model_id in [primary_model] + [m for m in fallback_models if m != primary_model]:
        try:
            response = client.responses.create(
                model=model_id,
                input=[
                    {"role": "system", "content": CONFIG.get("system_prompt", "")},
                    {"role": "user", "content": content},
                ],
            )
            return getattr(response, "output_text", str(response))
        except Exception as exc:
            last_err = exc
            msg = str(exc).lower()
            # Only fall through on auth/permission/model-not-found errors; other errors should surface.
            if not any(s in msg for s in ("model", "not found", "does not exist", "permission", "access", "404")):
                raise
            continue
    return (
        "⚠️ 사용 가능한 모델을 찾지 못했습니다 (시도: "
        f"{primary_model} + 폴백). 마지막 에러: {last_err}"
    )


def render_output_template(answer_text: str, contexts: list[dict[str, Any]], tool_results: list[str]) -> str:
    """Render the user-defined output_template with safe placeholder fill.

    Returns an empty string if no template is configured, in which case the caller should fall
    back to the raw answer text.
    """
    template = CONFIG.get("output_template") or ""
    if not template:
        return ""
    sources = format_sources(contexts)
    tool_text = "\n".join(tool_results) if tool_results else "도구 실행 없음"
    return safe_format(
        template,
        answer=answer_text,
        sources=sources,
        tool_results=tool_text,
        nutrition=tool_text,
        risks="(LLM 답변에서 추출되는 영역입니다)",
        next_steps="(필요한 추가 확인 항목)",
    )


def fallback_answer(message: str, session_id: str, image_b64: str | None, image_mime: str | None) -> dict[str, Any]:
    contexts = retrieve(
        message,
        strategy=CONFIG.get("retrieval_strategy", "hybrid_rerank"),
        k=int(CONFIG.get("top_k", 5)),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", CONFIG.get("embedding_model", "text-embedding-3-small")),
    )
    tool_results = maybe_run_tools(message)
    prompt = build_user_prompt(message, contexts, tool_results, image_attached=bool(image_b64), session_id=session_id)
    answer = call_gpt(prompt, image_b64=image_b64, image_mime=image_mime)
    remember(session_id, "user", message)
    remember(session_id, "assistant", answer)
    # GPT 답변이 이미 출력 템플릿 형식으로 작성되므로 추가 render는 생략 (가독성 ↑)
    payload = {
        "answer": answer,
        "formatted_answer": answer,
        "sources": format_sources(contexts),
        "tool_results": tool_results,
    }
    if CONFIG.get("show_debug_context"):
        payload["debug_contexts"] = contexts
    return payload


def answer_question(message: str, session_id: str = "default", image_b64: str | None = None, image_mime: str | None = None) -> dict[str, Any]:
    if CONFIG.get("enable_agent"):
        try:
            from agent_graph import invoke_graph
            return invoke_graph(message=message, session_id=session_id, image_b64=image_b64, image_mime=image_mime)
        except Exception:
            # Keep the deployed classroom bot resilient even if LangGraph dependency/setup fails.
            pass
    return fallback_answer(message, session_id, image_b64, image_mime)


# ----------------- Voice in/out (STT + answer + TTS) -----------------

STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")


def voice_chat(audio_bytes: bytes, audio_filename: str = "audio.webm", session_id: str = "default") -> dict[str, Any]:
    """Voice-in → text answer (using existing RAG/tool/Agent flow) → voice-out.

    Pipeline:
      1) STT: audio_bytes → transcript text (gpt-4o-mini-transcribe)
      2) answer_question(transcript): runs RAG + tools + Agent like a normal chat turn
      3) TTS: answer text → mp3 audio (gpt-4o-mini-tts, voice from CONFIG.voice_name)
    Returns a dict including base64-encoded mp3 the browser can play immediately.
    """
    import base64
    import io
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "error": "OPENAI_API_KEY가 설정되지 않았습니다. .env 또는 환경변수에 키를 설정하세요.",
        }
    try:
        from openai import OpenAI
    except Exception:
        return {"ok": False, "error": "openai 패키지가 없습니다."}
    client = OpenAI(api_key=api_key)

    # 1) STT
    try:
        transcription = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=(audio_filename, io.BytesIO(audio_bytes)),
        )
        transcript = (getattr(transcription, "text", "") or "").strip()
    except Exception as exc:
        return {"ok": False, "error": f"음성 인식 실패: {exc}"}

    if not transcript:
        return {"ok": False, "error": "음성을 인식하지 못했습니다. 다시 시도해주세요."}

    # 2) Existing answer flow (RAG + tools + Agent)
    result = answer_question(message=transcript, session_id=session_id)
    text_answer = result.get("formatted_answer") or result.get("answer") or "답변 생성 실패"

    # 3) TTS — cap at 4000 chars (TTS limit) and strip markdown that doesn't read well.
    import re as _re
    spoken_text = _re.sub(r"[#`*_>\[\]\(\)]+", " ", text_answer)
    spoken_text = _re.sub(r"\s+", " ", spoken_text).strip()[:4000]

    audio_b64 = ""
    audio_format = "mp3"
    voice = CONFIG.get("voice_name", "alloy")
    try:
        speech = client.audio.speech.create(
            model=TTS_MODEL,
            voice=voice,
            input=spoken_text,
            response_format=audio_format,
        )
        # SDK returns a streaming/bytes response. .read() works for HttpxBinaryResponseContent
        audio_bytes_out = speech.read() if hasattr(speech, "read") else speech.content
        audio_b64 = base64.b64encode(audio_bytes_out).decode("utf-8")
    except Exception as exc:
        # TTS failure still returns text so the UI can show something.
        return {
            "ok": True,
            "transcript": transcript,
            "text": text_answer,
            "audio_b64": "",
            "audio_format": audio_format,
            "voice": voice,
            "tts_error": str(exc),
            "sources": result.get("sources", ""),
            "tool_results": result.get("tool_results", []),
        }

    return {
        "ok": True,
        "transcript": transcript,
        "text": text_answer,
        "audio_b64": audio_b64,
        "audio_format": audio_format,
        "voice": voice,
        "sources": result.get("sources", ""),
        "tool_results": result.get("tool_results", []),
    }
