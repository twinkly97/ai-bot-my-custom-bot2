from __future__ import annotations

import ast
import operator as op
import os
import re
from typing import Any

import requests

SAFE_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod,
}


def safe_calculate(expression: str) -> str:
    def eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPS:
            return SAFE_OPS[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
            return SAFE_OPS[type(node.op)](eval_node(node.operand))
        raise ValueError("허용되지 않은 계산식입니다.")
    tree = ast.parse(expression, mode="eval")
    return str(eval_node(tree.body))


US_ALIASES = {
    "애플": "AAPL", "테슬라": "TSLA", "엔비디아": "NVDA",
    "마이크로소프트": "MSFT", "아마존": "AMZN", "구글": "GOOGL",
    "메타": "META", "넷플릭스": "NFLX", "아이비엠": "IBM", "엘비엠": "IBM",
    "AMD": "AMD", "에이엠디": "AMD",
    "팔란티어": "PLTR", "스타벅스": "SBUX", "맥도날드": "MCD",
    "코카콜라": "KO", "디즈니": "DIS",
}

# 한국 증시 — 코스피(.KS) + 코스닥(.KQ)
KR_ALIASES = {
    # 코스피 (KS)
    "삼성전자": "005930.KS", "삼성": "005930.KS",
    "SK하이닉스": "000660.KS", "하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS", "LG엔솔": "373220.KS",
    "삼성바이오로직스": "207940.KS", "삼바": "207940.KS",
    "현대차": "005380.KS", "현대자동차": "005380.KS",
    "기아": "000270.KS", "기아차": "000270.KS",
    "NAVER": "035420.KS", "네이버": "035420.KS",
    "카카오": "035720.KS",
    "셀트리온": "068270.KS",
    "POSCO홀딩스": "005490.KS", "포스코": "005490.KS", "포스코홀딩스": "005490.KS",
    "삼성SDI": "006400.KS",
    "LG화학": "051910.KS",
    "현대모비스": "012330.KS",
    "KB금융": "105560.KS", "KB금융지주": "105560.KS",
    "신한지주": "055550.KS", "신한금융": "055550.KS",
    "하나금융지주": "086790.KS", "하나금융": "086790.KS",
    "키움증권": "039490.KS", "키움": "039490.KS",
    "미래에셋증권": "006800.KS",
    "한국전력": "015760.KS", "한전": "015760.KS",
    "SK텔레콤": "017670.KS", "SKT": "017670.KS",
    "KT": "030200.KS",
    "LG전자": "066570.KS",
    "삼성생명": "032830.KS",
    "삼성화재": "000810.KS",
    "두산에너빌리티": "034020.KS",
    "한국조선해양": "009540.KS",
    "삼성중공업": "010140.KS",
    "대한항공": "003490.KS",
    "한진칼": "180640.KS",
    "이마트": "139480.KS",
    "신세계": "004170.KS",
    "롯데쇼핑": "023530.KS",
    "오리온": "271560.KS",
    "농심": "004370.KS",
    "CJ제일제당": "097950.KS",
    "한미약품": "128940.KS",
    "유한양행": "000100.KS",
    "녹십자": "006280.KS",
    "삼성SDS": "018260.KS", "삼성에스디에스": "018260.KS",
    # 코스닥 (KQ)
    "에코프로": "086520.KQ",
    "에코프로비엠": "247540.KQ",
    "엘앤에프": "066970.KQ",
    "알테오젠": "196170.KQ",
    "HLB": "028300.KQ",
    "리노공업": "058470.KQ",
    "JYP엔터": "035900.KQ", "JYP엔터테인먼트": "035900.KQ",
    "SK스퀘어": "402340.KQ",
    "솔브레인": "357780.KQ",
    "원익IPS": "240810.KQ",
    "씨젠": "096530.KQ",
    "에스엠": "041510.KQ", "SM엔터": "041510.KQ",
    "JYP": "035900.KQ",
    "하이브": "352820.KS",
    "와이지엔터": "122870.KQ", "YG엔터": "122870.KQ",
}


def extract_ticker(text: str) -> str | None:
    # 1) 이미 형식이 갖춰진 한국 티커 (.KS / .KQ)
    m = re.search(r"\b(\d{6})\.(KS|KQ)\b", text, re.IGNORECASE)
    if m:
        return f"{m.group(1)}.{m.group(2).upper()}"
    # 2) 6자리 숫자 → 기본 코스피 (.KS) — 강의용 충분
    m = re.search(r"\b(\d{6})\b", text)
    if m:
        return f"{m.group(1)}.KS"
    # 3) 미국 영문 티커 (1~5글자 대문자)
    candidates = re.findall(r"\b[A-Z]{1,5}\b", text.upper())
    stop = {"I", "A", "THE", "AND", "OR", "ETF", "EPS", "CEO", "CFO", "USA", "US", "GPT", "AI", "API", "RAG", "LLM"}
    for c in candidates:
        if c not in stop and c not in {"KS", "KQ"}:
            return c
    # 4) 한국어/한국 종목명 alias
    for name, ticker in KR_ALIASES.items():
        if name in text:
            return ticker
    # 5) 미국 영문/한글 alias
    for name, ticker in US_ALIASES.items():
        if name in text:
            return ticker
    return None


def stock_quote(symbol: str) -> dict[str, Any]:
    """Current price quote — RapidAPI 호환 또는 yfinance fallback."""
    symbol = symbol.upper().strip()
    api_key = os.getenv("YAHOO_FINANCE_API_KEY") or os.getenv("YAHOO_FINANCE_RAPIDAPI_KEY")
    host = os.getenv("YAHOO_FINANCE_RAPIDAPI_HOST", "apidojo-yahoo-finance-v1.p.rapidapi.com")
    if api_key:
        try:
            url = f"https://{host}/market/v2/get-quotes"
            params = {"region": "US", "symbols": symbol}
            headers = {"x-rapidapi-key": api_key, "x-rapidapi-host": host}
            res = requests.get(url, params=params, headers=headers, timeout=12)
            res.raise_for_status()
            data = res.json()
            quote = (data.get("quoteResponse") or {}).get("result", [{}])[0]
            return {
                "symbol": symbol,
                "provider": "rapidapi-yahoo-compatible",
                "price": quote.get("regularMarketPrice"),
                "currency": quote.get("currency"),
                "market_time": quote.get("regularMarketTime"),
                "raw_name": quote.get("shortName") or quote.get("longName"),
            }
        except Exception as exc:
            return {"symbol": symbol, "provider": "rapidapi-yahoo-compatible", "error": str(exc)}
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        hist = t.history(period="1d")
        price = None if hist.empty else float(hist["Close"].iloc[-1])
        info = getattr(t, "fast_info", {}) or {}
        return {"symbol": symbol, "provider": "yfinance-education-fallback", "price": price, "currency": info.get("currency")}
    except Exception as exc:
        return {"symbol": symbol, "provider": "none", "error": str(exc)}


# -----------------------------------------------------------------------------
# Financial-analyst-level tools (yfinance) — free, no API key needed.
# -----------------------------------------------------------------------------

def _yf_ticker(symbol: str):
    import yfinance as yf
    return yf.Ticker(symbol.upper().strip())


def get_company_info(symbol: str) -> dict[str, Any]:
    """회사 개요: 이름, 업종, 섹터, 본사, 임직원 수, 요약."""
    try:
        t = _yf_ticker(symbol)
        info = t.info or {}
        return {
            "symbol": symbol.upper(),
            "success": True,
            "company_name": info.get("longName") or info.get("shortName"),
            "industry": info.get("industry"),
            "sector": info.get("sector"),
            "country": info.get("country"),
            "website": info.get("website"),
            "employees": info.get("fullTimeEmployees"),
            "summary": (info.get("longBusinessSummary") or "")[:1200],
        }
    except Exception as exc:
        return {"symbol": symbol, "success": False, "error": str(exc)}


def get_financial_metrics(symbol: str) -> dict[str, Any]:
    """재무 지표: 시총, PER, PBR, ROE, 배당, 베타."""
    try:
        t = _yf_ticker(symbol)
        info = t.info or {}
        return {
            "symbol": symbol.upper(),
            "success": True,
            "market_cap_usd": info.get("marketCap"),
            "pe_trailing": info.get("trailingPE"),
            "pe_forward": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "return_on_equity": info.get("returnOnEquity"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception as exc:
        return {"symbol": symbol, "success": False, "error": str(exc)}


def get_stock_history(symbol: str, period: str = "1mo") -> dict[str, Any]:
    """기간별 주가 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max)."""
    valid = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
    if period not in valid:
        period = "1mo"
    try:
        t = _yf_ticker(symbol)
        hist = t.history(period=period)
        if hist.empty:
            return {"symbol": symbol, "success": False, "error": "데이터 없음"}
        first_close = float(hist["Close"].iloc[0])
        last_close = float(hist["Close"].iloc[-1])
        return {
            "symbol": symbol.upper(),
            "success": True,
            "period": period,
            "start_date": str(hist.index[0].date()),
            "end_date": str(hist.index[-1].date()),
            "start_close": first_close,
            "end_close": last_close,
            "change_pct": round((last_close - first_close) / first_close * 100, 2),
            "high": float(hist["High"].max()),
            "low": float(hist["Low"].min()),
            "volume_avg": int(hist["Volume"].mean()),
        }
    except Exception as exc:
        return {"symbol": symbol, "success": False, "error": str(exc)}


def get_income_statement(symbol: str) -> dict[str, Any]:
    """손익계산서 — 최근 4분기/연도."""
    try:
        t = _yf_ticker(symbol)
        df = t.income_stmt
        if df is None or df.empty:
            return {"symbol": symbol, "success": False, "error": "데이터 없음"}
        rows = {}
        for key in ("Total Revenue", "Gross Profit", "Operating Income", "Net Income", "Diluted EPS"):
            if key in df.index:
                series = df.loc[key].head(4)
                rows[key] = {str(c.date()) if hasattr(c, "date") else str(c): (float(v) if v == v else None) for c, v in series.items()}
        return {"symbol": symbol.upper(), "success": True, "income_statement_summary": rows}
    except Exception as exc:
        return {"symbol": symbol, "success": False, "error": str(exc)}


def get_balance_sheet(symbol: str) -> dict[str, Any]:
    """재무상태표 — 최근 4분기/연도."""
    try:
        t = _yf_ticker(symbol)
        df = t.balance_sheet
        if df is None or df.empty:
            return {"symbol": symbol, "success": False, "error": "데이터 없음"}
        rows = {}
        for key in ("Total Assets", "Total Liabilities Net Minority Interest", "Total Equity Gross Minority Interest", "Cash And Cash Equivalents", "Long Term Debt"):
            if key in df.index:
                series = df.loc[key].head(4)
                rows[key] = {str(c.date()) if hasattr(c, "date") else str(c): (float(v) if v == v else None) for c, v in series.items()}
        return {"symbol": symbol.upper(), "success": True, "balance_sheet_summary": rows}
    except Exception as exc:
        return {"symbol": symbol, "success": False, "error": str(exc)}


def get_cash_flow(symbol: str) -> dict[str, Any]:
    """현금흐름표 — 최근 4분기/연도."""
    try:
        t = _yf_ticker(symbol)
        df = t.cashflow
        if df is None or df.empty:
            return {"symbol": symbol, "success": False, "error": "데이터 없음"}
        rows = {}
        for key in ("Operating Cash Flow", "Free Cash Flow", "Capital Expenditure", "Issuance Of Debt", "Cash Dividends Paid"):
            if key in df.index:
                series = df.loc[key].head(4)
                rows[key] = {str(c.date()) if hasattr(c, "date") else str(c): (float(v) if v == v else None) for c, v in series.items()}
        return {"symbol": symbol.upper(), "success": True, "cash_flow_summary": rows}
    except Exception as exc:
        return {"symbol": symbol, "success": False, "error": str(exc)}


def _strip_html_tags(text: str) -> str:
    import html as _html
    cleaned = re.sub(r"<[^>]+>", "", text or "")
    return _html.unescape(cleaned).strip()


def naver_news_search(query: str, max_results: int = 5) -> list[dict[str, Any]] | None:
    """Naver Open API news search.

    Requires NAVER_CLIENT_ID and NAVER_CLIENT_SECRET environment variables.
    Returns None if either env var is missing → caller falls back to DuckDuckGo.

    API: https://openapi.naver.com/v1/search/news.json
    """
    client_id = os.getenv("NAVER_CLIENT_ID", "").strip()
    client_secret = os.getenv("NAVER_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return None
    try:
        res = requests.get(
            "https://openapi.naver.com/v1/search/news.json",
            headers={
                "X-Naver-Client-Id": client_id,
                "X-Naver-Client-Secret": client_secret,
            },
            params={
                "query": query,
                "display": str(min(max(max_results, 1), 100)),
                "start": "1",
                "sort": "date",
            },
            timeout=10,
        )
        res.raise_for_status()
        payload = res.json()
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in (payload.get("items") or []):
            url = item.get("originallink") or item.get("link") or ""
            if not url or url in seen:
                continue
            seen.add(url)
            results.append({
                "title": _strip_html_tags(item.get("title", "")),
                "url": url,
                "snippet": _strip_html_tags(item.get("description", ""))[:400],
                "published_date": item.get("pubDate"),
                "source": "naver_news",
            })
            if len(results) >= max_results:
                break
        return results
    except Exception as exc:
        # Naver failure is not fatal — caller should fall back
        return [{"error": str(exc), "query": query, "source": "naver_news"}]


def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """뉴스 검색. NAVER_CLIENT_ID/SECRET이 있으면 네이버 뉴스 우선 사용,
    없거나 실패하면 DuckDuckGo로 fallback."""
    # 1) Naver (instructor builder only — keys are not in student deployments)
    naver = naver_news_search(query, max_results=max_results)
    if naver and isinstance(naver, list) and naver and not naver[0].get("error"):
        return naver
    # 2) DuckDuckGo fallback
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        cleaned = []
        for r in results:
            cleaned.append({
                "title": r.get("title"),
                "url": r.get("href") or r.get("url"),
                "snippet": (r.get("body") or "")[:400],
                "source": "duckduckgo",
            })
        return cleaned
    except Exception as exc:
        return [{"error": str(exc), "query": query, "source": "duckduckgo"}]


def calorie_tool_notice() -> str:
    return "이미지 칼로리 추정은 GPT Vision 프롬프트로 처리합니다. 결과는 추정치이며 의료 조언이 아닙니다."


# -----------------------------------------------------------------------------
# 키움증권 REST API (국내 주식)
# 환경변수 KIWOOM_APP_KEY / KIWOOM_APP_SECRET / KIWOOM_BASE_URL 셋팅 시 동작.
# 기본 KIWOOM_BASE_URL은 모의투자 https://mockapi.kiwoom.com (실전은 https://api.kiwoom.com)
# -----------------------------------------------------------------------------
import time as _kw_time
from threading import Lock as _KwLock

_KIWOOM_TOKEN: dict[str, Any] = {"token": None, "expires_at": 0.0}
_KIWOOM_TOKEN_LOCK = _KwLock()


def _kiwoom_configured() -> bool:
    return bool(os.getenv("KIWOOM_APP_KEY") and os.getenv("KIWOOM_APP_SECRET"))


def _kiwoom_base_url() -> str:
    return os.getenv("KIWOOM_BASE_URL", "https://mockapi.kiwoom.com").rstrip("/")


def _kiwoom_get_token() -> str | None:
    """OAuth2 토큰 발급/캐싱. 만료 5분 전 자동 재발급."""
    if not _kiwoom_configured():
        return None
    now = _kw_time.time()
    with _KIWOOM_TOKEN_LOCK:
        cached = _KIWOOM_TOKEN.get("token")
        exp = _KIWOOM_TOKEN.get("expires_at", 0.0)
        if cached and now < (exp - 300):
            return cached
        try:
            res = requests.post(
                f"{_kiwoom_base_url()}/oauth2/token",
                json={
                    "grant_type": "client_credentials",
                    "appkey": os.getenv("KIWOOM_APP_KEY"),
                    "secretkey": os.getenv("KIWOOM_APP_SECRET"),
                },
                headers={"Content-Type": "application/json;charset=UTF-8"},
                timeout=10,
            )
            res.raise_for_status()
            data = res.json()
            token = data.get("token")
            expires_dt = data.get("expires_dt") or ""
            if not token:
                return None
            # expires_dt = YYYYMMDDHHMMSS (UTC) — parse roughly
            try:
                import datetime as _dt
                exp_ts = _dt.datetime.strptime(expires_dt, "%Y%m%d%H%M%S").replace(tzinfo=_dt.timezone.utc).timestamp()
            except Exception:
                exp_ts = now + 3600  # fallback: 1h
            _KIWOOM_TOKEN["token"] = token
            _KIWOOM_TOKEN["expires_at"] = exp_ts
            return token
        except Exception:
            return None


def _kiwoom_request(tr_id: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """저수준 키움 TR 요청. tr_id는 'ka10001' 같은 API ID."""
    if not _kiwoom_configured():
        return {"success": False, "error": "KIWOOM_APP_KEY/SECRET 환경변수가 설정되지 않음. 강사에게 문의."}
    token = _kiwoom_get_token()
    if not token:
        return {"success": False, "error": "키움 토큰 발급 실패. 키/네트워크 확인."}
    try:
        res = requests.post(
            f"{_kiwoom_base_url()}{path}",
            json=body or {},
            headers={
                "api-id": tr_id,
                "authorization": f"Bearer {token}",
                "Content-Type": "application/json;charset=UTF-8",
            },
            timeout=12,
        )
        res.raise_for_status()
        data = res.json()
        rc = str(data.get("return_code", "0"))
        if rc not in {"0", "00", "000", "0000"}:
            return {"success": False, "error": f"키움 return_code={rc}: {data.get('return_msg', '')}", "raw": data}
        return {"success": True, "data": data}
    except Exception as exc:
        return {"success": False, "error": f"키움 API 호출 실패: {exc}"}


def _kr_code_from_symbol(symbol: str) -> str | None:
    """`005930.KS`, `005930`, `삼성전자` 등에서 6자리 코드 추출."""
    s = (symbol or "").strip()
    if not s:
        return None
    # Korean name → code via alias map
    aliased = KR_ALIASES.get(s)
    if aliased:
        s = aliased
    # Strip .KS / .KQ suffix
    s = s.split(".")[0]
    # Validate: must be 6 digits
    if re.fullmatch(r"\d{6}", s):
        return s
    return None


def kiwoom_get_stock_info(symbol: str) -> dict[str, Any]:
    """키움 ka10001 — 주식 기본정보 (현재가, 등락률, 거래량, 시가/고가/저가, 시총).
    `symbol`은 6자리 코드, `.KS`/`.KQ` 붙은 형태, 또는 한국 종목명(삼성전자 등) 모두 가능."""
    code = _kr_code_from_symbol(symbol)
    if not code:
        return {"success": False, "error": f"한국 종목 코드를 인식하지 못했습니다: {symbol}"}
    res = _kiwoom_request("ka10001", "/api/dostk/stkinfo", {"stk_cd": code})
    if not res.get("success"):
        return {**res, "stock_code": code}
    d = res["data"]
    return {
        "success": True,
        "provider": "kiwoom",
        "stock_code": code,
        "stock_name": d.get("stk_nm"),
        "current_price": d.get("cur_prc"),
        "change": d.get("prdy_vrss"),
        "change_rate_pct": d.get("flu_rt"),
        "open": d.get("open_pric"),
        "high": d.get("high_pric"),
        "low": d.get("low_pric"),
        "volume": d.get("trde_qty"),
        "market_cap": d.get("mac"),
        "per": d.get("per"),
        "pbr": d.get("pbr"),
        "raw": d,
    }


def kiwoom_get_daily_chart(symbol: str, days: int = 30) -> dict[str, Any]:
    """키움 ka10081 — 일봉 차트. 기간 days(기본 30영업일)."""
    code = _kr_code_from_symbol(symbol)
    if not code:
        return {"success": False, "error": f"한국 종목 코드를 인식하지 못했습니다: {symbol}"}
    import datetime as _dt
    today = _dt.datetime.now().strftime("%Y%m%d")
    res = _kiwoom_request("ka10081", "/api/dostk/chart", {
        "stk_cd": code,
        "base_dt": today,
        "upd_stkpc_tp": "1",  # 수정주가 사용
    })
    if not res.get("success"):
        return {**res, "stock_code": code}
    d = res["data"]
    rows = (d.get("stk_dt_pole_chart_qry") or [])[:days]
    return {
        "success": True,
        "provider": "kiwoom",
        "stock_code": code,
        "rows": [
            {
                "date": r.get("dt"),
                "open": r.get("open_pric"),
                "high": r.get("high_pric"),
                "low": r.get("low_pric"),
                "close": r.get("cur_prc"),
                "volume": r.get("trde_qty"),
            }
            for r in rows
        ],
    }
