# 근퇴법쌤

근로자퇴직급여보장법에 대해 알기 쉽게 설명해주는 선생님봇

이 프로젝트는 AI Agent Builder Workshop에서 생성되었습니다. 업로드 문서는 `data/docs`, RAG 인덱스는 `data/index`, GraphRAG 아티팩트는 `data/graph`에 들어갑니다.

## 로컬 실행

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --reload --port 8080
```

브라우저:

```text
http://127.0.0.1:8080
```

## Vercel 배포

```bash
npm i -g vercel
vercel env add OPENAI_API_KEY production --sensitive
vercel env add YAHOO_FINANCE_API_KEY production --sensitive
vercel env add TELEGRAM_BOT_TOKEN production --sensitive
vercel --prod
```

## Telegram Webhook

Vercel 배포 URL이 `https://example.vercel.app`라면:

```bash
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \
  -d "url=https://example.vercel.app/telegram/webhook" \
  -d "secret_token=$TELEGRAM_WEBHOOK_SECRET"
```

## 보안

- `.env`는 커밋하지 마세요.
- API Key는 Vercel Environment Variables 또는 회사 Secret Manager에만 저장하세요.
- 업로드 문서에 개인정보/영업비밀이 포함되어 있으면 저장소 공개 여부를 반드시 확인하세요.


## 🚀 Streamlit Cloud 배포 (가장 쉬움 — 1분)

1. 이 ZIP을 풀고 GitHub repo로 push (private 권장)
2. https://share.streamlit.io 접속 → 본인 GitHub 계정 연동
3. 'New app' → 본인 repo 선택 → **Main file path: `streamlit_app.py`**
4. **Advanced settings → Secrets** 에 아래 내용 붙여넣기:
   ```toml
   OPENAI_API_KEY = "sk-..."
   YAHOO_FINANCE_API_KEY = ""
   TELEGRAM_BOT_TOKEN = ""
   ```
5. Deploy 클릭. 1분 뒤 공개 URL이 나옵니다.

## 로컬 실행

```bash
pip install -r requirements.txt
cp .env.example .env  # OPENAI_API_KEY 입력
streamlit run streamlit_app.py
```
