"""まちまど散歩エージェント（Cloud Run で動く Web サーバー）"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent import MAPS_API_KEY, run_walk
import voice


class JsonFormatter(logging.Formatter):
    """Cloud Logging が読める JSON 1行のログにする"""
    def format(self, r: logging.LogRecord) -> str:
        return json.dumps({"severity": r.levelname, "message": r.getMessage(), "logger": r.name}, ensure_ascii=False)


h = logging.StreamHandler(sys.stdout)
h.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[h])
log = logging.getLogger("machimado")

app = FastAPI(title="まちまど散歩エージェント")
HERE = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

# かんたんな使いすぎよけ（1つのIPから 1時間に RATE_PER_HOUR 回まで）
RATE_PER_HOUR = int(os.environ.get("RATE_PER_HOUR", "20"))
_hits: dict[str, deque] = defaultdict(deque)
VOICE_PER_HOUR = int(os.environ.get("VOICE_PER_HOUR", "200"))
_vhits: dict[str, deque] = defaultdict(deque)


def _over(store: dict, ip: str, limit: int) -> bool:
    now = time.time()
    q = store[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= limit:
        return True
    q.append(now)
    return False


def _client_ip(req: Request) -> str:
    return (req.headers.get("x-forwarded-for") or (req.client.host if req.client else "?")).split(",")[0].strip()


@app.get("/")
async def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"), headers={"Cache-Control": "no-store"})


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/api/config")
async def config():
    # 地図を表示するための鍵（Google Cloud 側で、使えるAPIとサイトを絞っておく）
    return {"mapsKey": MAPS_API_KEY}


@app.post("/api/voice")
async def speak(req: Request):
    """もちこの声（Gemini の音声生成）。失敗したら 502 を返し、画面側はブラウザの声に切りかえる"""
    if _over(_vhits, _client_ip(req), VOICE_PER_HOUR):
        return JSONResponse({"error": "rate"}, status_code=429)
    try:
        body = await req.json()
    except Exception:
        body = {}
    text = str(body.get("text", "")).strip()[:700]
    lang = "ja" if str(body.get("lang", "ja")).lower().startswith("ja") else "other"
    if not text:
        return JSONResponse({"error": "empty"}, status_code=400)
    t0 = time.time()
    try:
        wav = await voice.synth(text, lang)
    except Exception as e:
        log.warning(json.dumps({"event": "voice_error", "error": str(e)[:300]}, ensure_ascii=False))
        return JSONResponse({"error": "voice"}, status_code=502)
    log.info(json.dumps({"event": "voice", "chars": len(text), "seconds": round(time.time() - t0, 2), "model": voice._models[0] if voice._models else None}, ensure_ascii=False))
    return Response(wav, media_type="audio/wav", headers={"Cache-Control": "private, max-age=86400"})


@app.post("/api/walk")
async def walk(req: Request):
    ip = _client_ip(req)
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= RATE_PER_HOUR:
        return JSONResponse({"error": "少し時間をおいてから、もう一度お試しください。"}, status_code=429)
    q.append(now)
    try:
        body = await req.json()
    except Exception:
        body = {}
    text = str(body.get("request", "")).strip()[:300]
    if len(text) < 2:
        return JSONResponse({"error": "行きたい街と、どんな散歩にしたいかを書いてください。"}, status_code=400)
    trace_id = uuid.uuid4().hex[:12]
    log.info(json.dumps({"trace_id": trace_id, "event": "request", "request": text}, ensure_ascii=False))

    async def stream():
        async for ev in run_walk(text, trace_id):
            if ev.get("type") in ("done", "error"):
                log.info(json.dumps({"trace_id": trace_id, "event": ev["type"],
                                     "stats": (ev.get("walk") or {}).get("stats"), "message": ev.get("message")}, ensure_ascii=False))
            yield json.dumps(ev, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no", "X-Trace-Id": trace_id})
