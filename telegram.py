from __future__ import annotations

import base64
import os
from typing import Any

import requests

from bot_core import answer_question


def send_telegram_message(chat_id: int | str, text: str) -> dict[str, Any]:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN is not set"}
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text[:3900]}
    res = requests.post(url, json=payload, timeout=15)
    try:
        return res.json()
    except Exception:
        return {"ok": res.ok, "status_code": res.status_code, "text": res.text[:500]}


def _download_telegram_photo(file_id: str) -> tuple[str | None, str | None]:
    """Download the largest photo and return (b64, mime_guess)."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or not file_id:
        return None, None
    try:
        meta = requests.get(f"https://api.telegram.org/bot{token}/getFile", params={"file_id": file_id}, timeout=15).json()
        file_path = (meta.get("result") or {}).get("file_path")
        if not file_path:
            return None, None
        raw = requests.get(f"https://api.telegram.org/file/bot{token}/{file_path}", timeout=30).content
        mime = "image/jpeg" if file_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        return base64.b64encode(raw).decode("utf-8"), mime
    except Exception as exc:
        return None, f"download_failed:{type(exc).__name__}"


def handle_telegram_update(update: dict[str, Any]) -> dict[str, Any]:
    message = update.get("message") or update.get("edited_message") or {}
    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    text = message.get("text") or message.get("caption") or ""
    photos = message.get("photo") or []
    image_b64 = None
    image_mime = None
    if photos:
        # Telegram sends multiple sizes; pick the largest by file_size or last entry.
        largest = sorted(photos, key=lambda p: p.get("file_size", 0))[-1]
        image_b64, image_mime = _download_telegram_photo(largest.get("file_id", ""))
        if not text:
            text = "이 사진을 분석해줘."
    if not chat_id or (not text and not image_b64):
        return {"ok": True, "ignored": True}
    result = answer_question(text, session_id=f"telegram:{chat_id}", image_b64=image_b64, image_mime=image_mime)
    sent = send_telegram_message(chat_id, result.get("formatted_answer") or result.get("answer") or "답변 생성 실패")
    return {"ok": True, "telegram": sent}
