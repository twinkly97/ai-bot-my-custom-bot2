"""Telegram 봇 — long polling 방식 (webhook 불필요).

사용법:
1. Telegram에서 @BotFather 검색 → /newbot → 봇 이름·아이디 만들기 → 토큰 받기
2. .env 파일에 TELEGRAM_BOT_TOKEN=받은토큰 추가
3. python telegram_bot.py

특징:
- chat_id 단위로 대화 메모리 유지 (각 사용자마다 별도)
- 사진 업로드 시 GPT Vision으로 자동 분석 (enable_vision=True 프리셋에서)
- 식단 봇이면 누적 식단표 자동 생성 (시간 경과해도 표 갱신)
- Ctrl+C로 종료
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from bot_core import answer_question, CONFIG  # noqa: E402

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not TOKEN:
    print("❌ TELEGRAM_BOT_TOKEN 환경변수가 필요합니다.")
    print("   .env 파일에 TELEGRAM_BOT_TOKEN=받은토큰 추가하거나,")
    print("   PowerShell: $env:TELEGRAM_BOT_TOKEN='받은토큰'")
    sys.exit(1)

API_BASE = f"https://api.telegram.org/bot{TOKEN}"
BOT_NAME = CONFIG.get("bot_name", "AI Bot")
MAX_MSG_LEN = 4000  # Telegram 한 메시지 4096자 한도


def _send_message(chat_id: int, text: str, parse_mode: str | None = "Markdown") -> None:
    """Telegram으로 메시지 보내기. Markdown 파싱 실패 시 평문으로 재시도."""
    if not text:
        return
    chunks = [text[i : i + MAX_MSG_LEN] for i in range(0, len(text), MAX_MSG_LEN)] or [text]
    for chunk in chunks:
        payload = {"chat_id": chat_id, "text": chunk}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            r = requests.post(f"{API_BASE}/sendMessage", json=payload, timeout=10)
            if not r.ok and parse_mode:
                # Markdown 오류 → 평문으로 재시도
                payload.pop("parse_mode", None)
                requests.post(f"{API_BASE}/sendMessage", json=payload, timeout=10)
        except Exception as exc:
            print(f"[send_message] 실패: {exc}")


def _send_chat_action(chat_id: int, action: str = "typing") -> None:
    """봇이 응답 준비 중임을 표시 (타이핑 점점점)."""
    try:
        requests.post(f"{API_BASE}/sendChatAction", json={"chat_id": chat_id, "action": action}, timeout=5)
    except Exception:
        pass


def _download_photo(file_id: str) -> tuple[str | None, str]:
    """photo file_id → base64 이미지."""
    try:
        info = requests.get(f"{API_BASE}/getFile", params={"file_id": file_id}, timeout=10).json()
        if not info.get("ok"):
            return None, "image/jpeg"
        file_path = info["result"]["file_path"]
        file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
        raw = requests.get(file_url, timeout=30).content
        b64 = base64.b64encode(raw).decode("utf-8")
        mime = "image/jpeg" if file_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        return b64, mime
    except Exception as exc:
        print(f"[download_photo] 실패: {exc}")
        return None, "image/jpeg"


def _handle_update(update: dict) -> None:
    """단일 Telegram update 처리."""
    msg = update.get("message") or update.get("edited_message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    if not chat_id:
        return

    session_id = f"tg:{chat_id}"
    user_name = (msg.get("from") or {}).get("first_name", "")

    text = (msg.get("text") or msg.get("caption") or "").strip()
    photos = msg.get("photo") or []
    image_b64 = None
    image_mime = "image/png"

    # /start, /help 등 명령 처리
    if text == "/start":
        _send_message(chat_id, f"안녕하세요 {user_name}님! 🤖\n저는 *{BOT_NAME}*입니다.\n\n사진 또는 메시지를 보내주세요.")
        return
    if text == "/reset":
        # 메모리 초기화 (다음 답변부터 새 세션으로)
        try:
            from bot_core import _MEMORY  # noqa
            _MEMORY.pop(session_id, None)
            _send_message(chat_id, "🧹 대화 기록을 초기화했습니다. 새로 시작하세요!")
        except Exception:
            _send_message(chat_id, "초기화 실패")
        return

    # 사진 있으면 가장 큰 해상도 받기
    if photos:
        largest = max(photos, key=lambda p: p.get("file_size", 0))
        image_b64, image_mime = _download_photo(largest["file_id"])
        if not text:
            # 사진만 있고 메시지 없을 때 기본 프롬프트
            if (CONFIG.get("preset") or "").lower() == "diet" or CONFIG.get("enable_calorie_tool"):
                text = "이 음식의 칼로리와 영양소를 추정하고, 오늘 누적 표에 추가해줘."
            else:
                text = "이 사진을 분석해줘."

    if not text and not image_b64:
        return  # 비어있는 메시지 무시

    _send_chat_action(chat_id, "typing")
    print(f"[{chat_id}] {user_name}: {text[:100]}{'... + 📷' if image_b64 else ''}")

    try:
        result = answer_question(
            message=text,
            session_id=session_id,
            image_b64=image_b64,
            image_mime=image_mime,
        )
        answer = result.get("formatted_answer") or result.get("answer") or "❌ 답변 생성 실패"
        _send_message(chat_id, answer)
    except Exception as exc:
        print(f"[handle_update] 답변 실패: {exc}")
        _send_message(chat_id, f"❌ 처리 중 오류: {str(exc)[:200]}", parse_mode=None)


def main() -> None:
    print(f"🤖 [{BOT_NAME}] Telegram 봇 시작 (long polling)")
    print(f"   토큰 확인: {TOKEN[:10]}...{TOKEN[-4:]}")
    print(f"   봇에게 메시지를 보내려면 Telegram에서 봇 검색 후 /start")
    print(f"   종료: Ctrl+C\n")

    offset = None
    while True:
        try:
            resp = requests.get(
                f"{API_BASE}/getUpdates",
                params={"offset": offset, "timeout": 30},
                timeout=35,
            ).json()
            if not resp.get("ok"):
                print(f"[getUpdates] {resp}")
                time.sleep(3)
                continue
            for update in resp.get("result", []):
                offset = update["update_id"] + 1
                try:
                    _handle_update(update)
                except Exception as exc:
                    print(f"[update {update.get('update_id')}] 처리 실패: {exc}")
        except KeyboardInterrupt:
            print("\n👋 종료합니다")
            break
        except requests.RequestException as exc:
            print(f"[network] {exc}, 5초 후 재시도")
            time.sleep(5)
        except Exception as exc:
            print(f"[loop] {exc}")
            time.sleep(3)


if __name__ == "__main__":
    main()
