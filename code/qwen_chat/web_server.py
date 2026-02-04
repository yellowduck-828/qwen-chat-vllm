# pyright: reportMissingImports=false
"""
FastAPI-based proxy for remote vLLM.
If your editor reports missing FastAPI imports, ensure the interpreter
has FastAPI installed; missing-import checks are disabled here to avoid
false alarms in lint-only environments.
"""
import os
import json
import requests
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Dict
from pathlib import Path
import threading
import time

app = FastAPI()

# ===============================
# 配置 vLLM 后端地址（关键修复）
# ===============================
REMOTE_VLLM_BASE_URL = os.getenv("REMOTE_VLLM_BASE_URL", "http://127.0.0.1:7000")
REMOTE_VLLM_MODEL = os.getenv("REMOTE_VLLM_MODEL", "Qwen3-8B")
# vLLM 请求超时（秒），避免长时间无响应卡住
REMOTE_VLLM_TIMEOUT = float(os.getenv("REMOTE_VLLM_TIMEOUT", "30"))

print(">>> Using remote vLLM backend:", REMOTE_VLLM_BASE_URL)
print(">>> Using model:", REMOTE_VLLM_MODEL)

# CORS 允许跨域访问（前端网页需要）
# 可通过环境变量 ALLOWED_ORIGINS="https://foo.com,https://bar.com" 覆盖
_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()]
ALLOW_ALL_ORIGINS = ALLOWED_ORIGINS == ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ALL_ORIGINS else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
#   前端网页静态资源
# ===============================
STATIC_DIR = "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.get("/")
async def root_page():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return JSONResponse(status_code=404, content={"error": "index.html not found"})
    return FileResponse(index_path)


# ===============================
#   Chat 请求数据结构
# ===============================
class ChatRequest(BaseModel):
    chat_id: str
    new_message: str
    enable_thinking: bool | None = None
    stream: bool | None = None


class HistoryResponse(BaseModel):
    chat_id: str
    session: dict
    messages: list | None = None
    recent_messages: list | None = None
    memory_keywords: list | None = None


def _extract_reply_text(choice: dict) -> str:
    """Extract visible text; if正文缺失则回退到 reasoning 内容。"""
    msg = choice.get("message") or {}

    def _join_visible(value):
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = []
            for chunk in value:
                if isinstance(chunk, str):
                    parts.append(chunk)
                elif isinstance(chunk, dict):
                    # keep only plain text chunks
                    if chunk.get("type") == "text" and "text" in chunk:
                        parts.append(chunk["text"])
            return "".join(parts).strip()
        return ""

    # Prefer message.content (text only)
    content = _join_visible(msg.get("content"))
    # Then text fields
    text_field = _join_visible(choice.get("text") or msg.get("text"))
    # Then alternative content field
    alt_content = _join_visible(choice.get("content"))
    # Fallback: reasoning content when no visible text
    def _join_reasoning(value):
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = []
            for chunk in value:
                if isinstance(chunk, str):
                    parts.append(chunk)
                elif isinstance(chunk, dict):
                    # 常见结构: {"type": "reasoning", "reasoning": "..."}
                    if chunk.get("type") == "reasoning" and "reasoning" in chunk:
                        parts.append(chunk["reasoning"])
            return "".join(parts).strip()
        return ""

    reasoning = _join_reasoning(msg.get("reasoning_content") or choice.get("reasoning_content"))

    for cand in (content, text_field, alt_content, reasoning):
        if cand:
            return cand
    return ""


def _extract_reasoning(choice: dict) -> str:
    """Extract reasoning_content when available."""
    msg = choice.get("message") or {}

    def _join_reasoning(value):
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = []
            for chunk in value:
                if isinstance(chunk, str):
                    parts.append(chunk)
                elif isinstance(chunk, dict):
                    if chunk.get("type") == "reasoning" and "reasoning" in chunk:
                        parts.append(chunk["reasoning"])
                    elif "text" in chunk:
                        parts.append(str(chunk["text"]))
            return "".join(parts).strip()
        return ""

    return _join_reasoning(msg.get("reasoning_content") or choice.get("reasoning_content"))


# ===============================
#   简单的后端会话存储（内存 + 本地文件持久化）
# ===============================
CHAT_SESSIONS: Dict[str, dict] = {}
# 最近轮次条数：2 轮=4 条
RECENT_ROUNDS = 2
RECENT_MESSAGES_LIMIT = RECENT_ROUNDS * 2
# 关键词记忆上限，避免无限增长
MAX_MEMORY_KEYWORDS = 50
# 会话持久化文件（可通过环境变量覆盖）
# 默认固定到工作区源码目录，避免运行目录变化导致存储跑偏
_default_sessions_path = Path("/root/autodl-fs/code/qwen_chat/chat_sessions.json")
SESSIONS_FILE = Path(os.getenv("CHAT_SESSIONS_FILE", _default_sessions_path))
_SAVE_LOCK = threading.Lock()


def _create_empty_session():
    return {"recent_messages": [], "memory_keywords": [], "meta": {"topic": "上下文管理", "last_updated": int(time.time())}}


def _message_to_keyword(msg: dict, max_len: int = 60) -> str:
    """将单条消息压缩为关键词描述。"""
    if not isinstance(msg, dict):
        return ""
    role = msg.get("role", "user")
    content = (msg.get("content") or "").replace("\n", " ").strip()
    if not content:
        return ""
    snippet = content[:max_len]
    return f"{role}: {snippet}"


def _normalize_session_obj(raw) -> dict:
    """兼容旧格式(list)并补全缺省字段。"""
    session = _create_empty_session()
    if isinstance(raw, dict):
        session["recent_messages"] = list(raw.get("recent_messages") or [])
        session["memory_keywords"] = list(raw.get("memory_keywords") or [])
        meta = dict(raw.get("meta") or {})
        meta.setdefault("topic", "上下文管理")
        meta.setdefault("last_updated", int(time.time()))
        session["meta"] = meta
    elif isinstance(raw, list):
        # 旧格式：纯消息列表，迁移为 recent + keywords
        if raw:
            recent = raw[-RECENT_MESSAGES_LIMIT:]
            older = raw[:-RECENT_MESSAGES_LIMIT]
            keywords = [_message_to_keyword(m) for m in older if _message_to_keyword(m)]
            session["recent_messages"] = recent
            session["memory_keywords"] = keywords[-MAX_MEMORY_KEYWORDS:]
    _cap_memory_keywords(session)
    return session


def _load_sessions_from_disk():
    if not SESSIONS_FILE.exists():
        print(f">>> Session file not found, will create: {SESSIONS_FILE}")
        return
    try:
        data = json.loads(SESSIONS_FILE.read_text("utf-8"))
        if isinstance(data, dict):
            for k, v in data.items():
                CHAT_SESSIONS[k] = _normalize_session_obj(v)
        print(f">>> Loaded {len(CHAT_SESSIONS)} chat sessions from {SESSIONS_FILE}")
    except Exception as e:
        print(">>> Failed to load sessions file:", e)


def _save_sessions_to_disk():
    with _SAVE_LOCK:
        try:
            tmp_path = SESSIONS_FILE.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(CHAT_SESSIONS, ensure_ascii=False), "utf-8")
            tmp_path.replace(SESSIONS_FILE)
            print(f">>> Saved {len(CHAT_SESSIONS)} sessions to {SESSIONS_FILE}")
        except Exception as e:
            print(">>> Failed to save sessions file:", e)


def _get_session(chat_id: str) -> dict:
    if chat_id not in CHAT_SESSIONS:
        CHAT_SESSIONS[chat_id] = _create_empty_session()
        _save_sessions_to_disk()
    else:
        CHAT_SESSIONS[chat_id] = _normalize_session_obj(CHAT_SESSIONS[chat_id])
    return CHAT_SESSIONS[chat_id]


def _cap_memory_keywords(session: dict):
    if len(session.get("memory_keywords", [])) > MAX_MEMORY_KEYWORDS:
        session["memory_keywords"] = session["memory_keywords"][-MAX_MEMORY_KEYWORDS:]


def _append_message(chat_id: str, message: dict):
    """写入消息到 recent_messages，并将超出部分压缩到 memory_keywords。"""
    session = _get_session(chat_id)
    session["recent_messages"].append(message)
    while len(session["recent_messages"]) > RECENT_MESSAGES_LIMIT:
        oldest = session["recent_messages"].pop(0)
        kw = _message_to_keyword(oldest)
        if kw:
            session["memory_keywords"].append(kw)
    _cap_memory_keywords(session)
    session["meta"]["last_updated"] = int(time.time())
    CHAT_SESSIONS[chat_id] = session
    _save_sessions_to_disk()


def _build_memory_system_prompt(session: dict):
    """将关键词转为 system 提示，帮助模型保持上下文。"""
    keywords = session.get("memory_keywords") or []
    if not keywords:
        return None
    content = "Earlier conversation keywords:\n" + "\n".join(f"- {kw}" for kw in keywords)
    return {"role": "system", "content": content}


def _delete_session(chat_id: str) -> bool:
    """删除指定会话，内存+磁盘同步"""
    existed = CHAT_SESSIONS.pop(chat_id, None) is not None
    if existed:
        _save_sessions_to_disk()
    return existed


# 加载历史会话（持久化）
_load_sessions_from_disk()


# ===============================
#   配置与健康检查
# ===============================
@app.get("/api/config")
async def config():
    return {"backend": REMOTE_VLLM_BASE_URL, "model": REMOTE_VLLM_MODEL}


@app.get("/api/health")
async def health():
    return {"status": "ok", "backend": REMOTE_VLLM_BASE_URL, "model": REMOTE_VLLM_MODEL}


@app.get("/api/history/{chat_id}", response_model=HistoryResponse)
async def history(chat_id: str):
    session_state = _get_session(chat_id)
    return {
        "chat_id": chat_id,
        "session": session_state,
        # 兼容旧前端：直接返回 recent 与 keywords
        "messages": session_state.get("recent_messages", []),
        "recent_messages": session_state.get("recent_messages", []),
        "memory_keywords": session_state.get("memory_keywords", []),
    }


@app.delete("/api/history/{chat_id}")
async def delete_history(chat_id: str):
    if _delete_session(chat_id):
        return {"chat_id": chat_id, "deleted": True}
    return JSONResponse(status_code=404, content={"error": "chat_id not found"})


# ===============================
#   /v1/chat/completions  → 转发到 vLLM
# ===============================
@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    # 组装 messages，后端负责管理历史
    base_system = {"role": "system", "content": "You are a helpful assistant."}
    user_msg = {"role": "user", "content": req.new_message}
    session_state = _get_session(req.chat_id)
    memory_prompt = _build_memory_system_prompt(session_state)
    history_msgs = list(session_state.get("recent_messages", []))
    msgs = [base_system]
    if memory_prompt:
        msgs.append(memory_prompt)
    msgs += history_msgs + [user_msg]
    # 先写入用户消息，确保即便后续出错也保留上下文
    _append_message(req.chat_id, user_msg)
    print(
        f">>> chat_id={req.chat_id} history_len(after user)={len(_get_session(req.chat_id).get('recent_messages', []))}"
    )

    # shared payload
    thinking_enabled = bool(req.enable_thinking)

    payload = {
        "model": REMOTE_VLLM_MODEL,
        "messages": msgs,
        # 增大生成上限，避免长答复被截断
        "max_tokens": 2048,
        "temperature": 0.7,
    }
    # 可选启用思考模式（思维链）
    if req.enable_thinking is not None:
        payload["enable_thinking"] = req.enable_thinking

    vllm_url = f"{REMOTE_VLLM_BASE_URL}/v1/chat/completions"
    print(">>> Calling vLLM:", vllm_url, "stream=", bool(req.stream))

    # ---------- Stream mode ----------
    if req.stream:
        async def event_gen():
            stream_payload = dict(payload)
            stream_payload["stream"] = True
            answer_buffer = []
            reasoning_buffer = []
            line_count = 0
            try:
                async with httpx.AsyncClient(timeout=REMOTE_VLLM_TIMEOUT) as client:
                    async with client.stream("POST", vllm_url, json=stream_payload) as r:
                        async for line in r.aiter_lines():
                            if not line:
                                continue
                            line_count += 1
                            if line_count <= 3:
                                print("<<< stream line:", line[:200])
                            if line.startswith("data:"):
                                line = line[len("data:") :].strip()
                            if line == "[DONE]":
                                # 不要提前 return，确保后续保存历史
                                yield "data: [DONE]\n\n"
                                break
                            try:
                                obj = json.loads(line)
                            except Exception:
                                # 不可解析的行忽略
                                continue

                            choice = (obj.get("choices") or [{}])[0]
                            delta = choice.get("delta") or {}

                            def _join_content(value):
                                if isinstance(value, str):
                                    return value
                                if isinstance(value, list):
                                    parts = []
                                    for chunk in value:
                                        if isinstance(chunk, str):
                                            parts.append(chunk)
                                        elif isinstance(chunk, dict):
                                            if chunk.get("type") == "text" and "text" in chunk:
                                                parts.append(chunk["text"])
                                            elif "reasoning" in chunk:
                                                parts.append(chunk["reasoning"])
                                    return "".join(parts)
                                if isinstance(value, dict):
                                    # 兼容 {"reasoning": "..."} 结构
                                    if "reasoning" in value:
                                        return value["reasoning"]
                                    if "text" in value:
                                        return value["text"]
                                return ""

                            delta_text = _join_content(delta.get("content"))
                            # vLLM 可能同时返回 reasoning 或 reasoning_content
                            delta_reasoning = _join_content(
                                delta.get("reasoning_content") or delta.get("reasoning")
                            )

                            # 只有思考模式开启时才处理 reasoning；否则忽略
                            if not thinking_enabled:
                                delta_reasoning = ""

                            out = {}
                            if delta_text:
                                out["delta"] = delta_text
                                answer_buffer.append(delta_text)
                            if delta_reasoning:
                                out["delta_reasoning"] = delta_reasoning
                                reasoning_buffer.append(delta_reasoning)
                            if out:
                                yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            except Exception as e:
                err = {"error": f"stream error: {e}"}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            # 保存历史（注意不要在接收到 [DONE] 时提前 return）
            if answer_buffer or reasoning_buffer:
                # 保存助手回复，并在启用思考时携带 reasoning 字段
                content = "".join(answer_buffer)

                _append_message(req.chat_id, {"role": "assistant", "content": content})
                print(
                    f">>> chat_id={req.chat_id} saved stream reply, "
                    f"recent_len={len(_get_session(req.chat_id).get('recent_messages', []))}, user+assistant appended"
                )

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    # ---------- Non-stream mode ----------
    try:
        response = requests.post(vllm_url, json=payload, timeout=REMOTE_VLLM_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            text = _extract_reply_text(choice)
            thinking = _extract_reasoning(choice) if thinking_enabled else ""
            if text:
                resp = {"reply": text}
                if thinking:
                    resp["thinking"] = thinking
                # 保存历史
                _append_message(req.chat_id, {"role": "assistant", "content": text})
                print(
                    f">>> chat_id={req.chat_id} saved non-stream reply, "
                    f"recent_len={len(_get_session(req.chat_id).get('recent_messages', []))}"
                )
                return resp
            # 若正文为空且启用思考，再用思考作为可见回复
            if thinking_enabled and thinking:
                _append_message(req.chat_id, {"role": "assistant", "content": thinking})
                print(
                    f">>> chat_id={req.chat_id} saved non-stream reasoning-as-reply, "
                    f"recent_len={len(_get_session(req.chat_id).get('recent_messages', []))}"
                )
                return {"reply": thinking, "thinking": thinking}
            return JSONResponse(
                status_code=500,
                content={"error": "Empty reply from backend", "detail": data},
            )

        return JSONResponse(status_code=500, content={"error": "Invalid vLLM response", "detail": data})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Exception calling vLLM", "detail": str(e)})


# ===============================
#   启动方式（如果直接 python web_server.py）
# ===============================
if __name__ == "__main__":
    import uvicorn
    WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
    WEB_PORT = int(os.getenv("WEB_PORT", 6006))

    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT)
