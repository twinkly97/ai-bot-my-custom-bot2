"""Streamlit frontend for the generated chatbot.

Deploy this to Streamlit Community Cloud:
1. Push this repo to GitHub.
2. Visit https://share.streamlit.io and connect the repo.
3. Set main file to `streamlit_app.py`.
4. In "Advanced settings → Secrets", paste your OPENAI_API_KEY (and optional YAHOO_FINANCE_API_KEY).
5. Deploy. Streamlit will give you a public URL.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import streamlit as st

# Load Streamlit secrets into environment so bot_core can read them as it would in any deploy target.
try:
    for k, v in st.secrets.items():
        if isinstance(v, str) and not os.getenv(k):
            os.environ[k] = v
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))

from bot_core import answer_question, CONFIG  # noqa: E402


st.set_page_config(page_title=CONFIG.get("bot_name", "AI Chatbot"), page_icon="🤖", layout="centered")
st.title(CONFIG.get("bot_name", "AI Chatbot"))
description = CONFIG.get("description", "")
if description:
    st.caption(description)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = f"streamlit:{uuid.uuid4().hex[:12]}"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

vision_enabled = bool(CONFIG.get("enable_vision"))
image_b64 = None
image_mime = None
if vision_enabled:
    image_file = st.file_uploader("이미지 업로드 (선택)", type=["png", "jpg", "jpeg"])
    if image_file:
        raw = image_file.read()
        image_b64 = base64.b64encode(raw).decode("utf-8")
        image_mime = image_file.type or "image/png"
        st.image(raw, caption="첨부된 이미지", width=240)

# 함수 먼저 정의 — 아래 voice 블록이 사용하기 전에 선언돼야 함 (NameError 방지)
def bot_core_voice_chat(audio_bytes: bytes, session_id: str):
    """Wrapper so the Streamlit code above can call bot_core.voice_chat with a clean signature."""
    from bot_core import voice_chat as _vc
    return _vc(audio_bytes=audio_bytes, audio_filename="audio.webm", session_id=session_id)


# 🎙️ Voice input (only when the bot was generated with enable_voice=True)
if CONFIG.get("enable_voice"):
    try:
        from streamlit_mic_recorder import mic_recorder
    except Exception:
        mic_recorder = None
    if mic_recorder is not None:
        st.caption("🎙️ 음성으로 질문하기")
        audio = mic_recorder(start_prompt="🎤 녹음 시작", stop_prompt="⏹ 녹음 중지", just_once=True, key="voice_in")
        if audio and audio.get("bytes"):
            with st.chat_message("user"):
                st.markdown("🎤 음성 입력")
            with st.chat_message("assistant"):
                with st.spinner("음성 인식 + 답변 + 음성 합성 중..."):
                    result = bot_core_voice_chat(audio["bytes"], st.session_state.session_id)
                if result.get("ok"):
                    if result.get("transcript"):
                        st.caption(f"📝 인식된 질문: {result['transcript']}")
                    st.markdown(result.get("text", ""))
                    if result.get("audio_b64"):
                        import base64 as _b64
                        st.audio(_b64.b64decode(result["audio_b64"]), format=f"audio/{result.get('audio_format', 'mp3')}", autoplay=True)
                    st.session_state.messages.append({"role": "user", "content": "🎤 " + (result.get("transcript") or "음성 입력")})
                    st.session_state.messages.append({"role": "assistant", "content": result.get("text", "")})
                else:
                    st.error(result.get("error", "음성 응답 실패"))
    else:
        st.info("음성 기능을 사용하려면 `pip install streamlit-mic-recorder`를 추가하세요.")


prompt = st.chat_input("메시지를 입력하세요…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("생각 중…"):
            result = answer_question(
                message=prompt,
                session_id=st.session_state.session_id,
                image_b64=image_b64,
                image_mime=image_mime,
            )
        answer = result.get("formatted_answer") or result.get("answer") or "답변 생성 실패"
        st.markdown(answer)
        if result.get("tool_results"):
            with st.expander("🔧 도구 결과"):
                for r in result["tool_results"]:
                    st.code(r)
        if result.get("sources") and result["sources"] != "검색된 문서 근거 없음":
            with st.expander("📚 RAG 검색 근거"):
                st.markdown(result["sources"])
        st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("챗봇 설정")
    st.write(f"**모델:** {CONFIG.get('model_name')}")
    st.write(f"**검색 방식:** {CONFIG.get('retrieval_strategy')}")
    st.write(f"**메모리:** {'ON' if CONFIG.get('use_memory') else 'OFF'} (최근 {CONFIG.get('memory_turns')}턴)")
    st.write(f"**GraphRAG:** {'ON' if CONFIG.get('enable_graphrag') else 'OFF'}")
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()
    with st.expander("⚙️ 시스템 프롬프트 보기"):
        st.code(CONFIG.get("system_prompt", ""), language="markdown")

    # ────────────────────────────────────────────────────────────────
    # 📚 문서 관리 (배포 후에도 문서 추가·삭제·재인덱싱 가능)
    # ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📚 문서 관리")

    try:
        from reindex import list_documents, save_uploaded_file, delete_document, rebuild_index, ALLOWED_EXTENSIONS
    except ImportError:
        st.caption("⚠️ reindex 모듈을 찾지 못했습니다.")
    else:
        DOCS_DIR = ROOT / "data" / "docs"
        INDEX_DIR = ROOT / "data" / "index"

        current_docs = list_documents(DOCS_DIR)
        if current_docs:
            st.caption(f"현재 인덱싱된 문서 ({len(current_docs)}개)")
            for doc in current_docs:
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.write(f"📄 {doc['name']} _({doc['size_kb']}KB)_")
                with col2:
                    if st.button("🗑️", key=f"del_{doc['name']}", help="이 문서 삭제 (재인덱싱은 별도로 눌러야 적용)"):
                        delete_document(doc["name"], DOCS_DIR)
                        st.toast(f"✓ 삭제됨: {doc['name']}")
                        st.rerun()
        else:
            st.caption("아직 인덱싱된 문서가 없습니다.")

        st.write("**➕ 새 문서 추가**")
        allowed_str = ", ".join(sorted(ext.lstrip(".") for ext in ALLOWED_EXTENSIONS))
        uploaded = st.file_uploader(
            "파일 업로드",
            type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
            accept_multiple_files=True,
            help=f"지원: {allowed_str}",
            key="doc_uploader",
        )
        if uploaded:
            if st.button(f"📥 {len(uploaded)}개 파일 docs 폴더에 저장"):
                ok_count = 0
                for f in uploaded:
                    try:
                        save_uploaded_file(f, f.name, DOCS_DIR)
                        ok_count += 1
                    except Exception as exc:
                        st.error(f"{f.name}: {exc}")
                if ok_count:
                    st.toast(f"✓ {ok_count}개 저장됨 — 아래 「🔄 인덱스 재빌드」를 눌러 적용하세요")
                    st.rerun()

        st.write("**🔄 인덱스 재빌드**")
        col_a, col_b = st.columns(2)
        with col_a:
            re_chunk_size = st.number_input("chunk_size", 200, 4000, int(CONFIG.get("chunk_size", 900)), 50)
        with col_b:
            re_chunk_overlap = st.number_input("overlap", 0, 1000, int(CONFIG.get("chunk_overlap", 120)), 10)
        if st.button("🔄 인덱스 재빌드 실행", type="primary", help="docs 폴더의 모든 파일로 FAISS 인덱스를 다시 만듭니다 (수십초 소요)"):
            with st.spinner("재인덱싱 중... (문서 양에 따라 10~60초)"):
                result = rebuild_index(
                    docs_dir=DOCS_DIR,
                    index_dir=INDEX_DIR,
                    chunk_size=int(re_chunk_size),
                    chunk_overlap=int(re_chunk_overlap),
                    embedding_model=CONFIG.get("embedding_model", "text-embedding-3-small"),
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                )
            if result.get("ok"):
                st.success(result.get("message") or "재인덱싱 완료")
                # Drop cached retrieval state so next query uses fresh index
                st.session_state.messages = []
                st.rerun()
            else:
                st.error(result.get("error") or "재인덱싱 실패")

        st.caption("ℹ️ Streamlit Cloud는 디스크가 휘발성이라 앱 재시작 시 변경된 문서가 초기화될 수 있습니다. 영구 유지하려면 GitHub repo에 직접 커밋하세요.")
