"""まちまど散歩エージェント（Cloud Run で動く Web サーバー）"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid

import httpx
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


@app.get("/voice")
async def voice_page():
    return FileResponse(os.path.join(HERE, "static", "voice.html"), headers={"Cache-Control": "no-store"})


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/api/config")
async def config():
    # 地図を表示するための鍵（Google Cloud 側で、使えるAPIとサイトを絞っておく）
    return {"mapsKey": MAPS_API_KEY}


_local_cache: dict[str, tuple[float, dict]] = {}
_WMO = {0: "快晴", 1: "晴れ", 2: "くもり時々晴れ", 3: "くもり", 45: "霧", 48: "霧", 51: "霧雨", 53: "霧雨", 55: "霧雨",
        61: "小雨", 63: "雨", 65: "強い雨", 66: "冷たい雨", 67: "冷たい雨", 71: "小雪", 73: "雪", 75: "大雪", 77: "雪",
        80: "にわか雨", 81: "にわか雨", 82: "強いにわか雨", 85: "にわか雪", 86: "にわか雪", 95: "雷雨", 96: "雷雨", 99: "雷雨"}


@app.get("/api/local")
async def local_info(lat: float, lng: float):
    """現地の時間（タイムゾーン）と、いまの天気。Google の Weather API、だめなら Open-Meteo"""
    key = f"{lat:.2f},{lng:.2f}"
    hit = _local_cache.get(key)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    out: dict = {}
    async with httpx.AsyncClient(timeout=8.0) as http:
        try:
            r = await http.get("https://weather.googleapis.com/v1/currentConditions:lookup", params={
                "key": MAPS_API_KEY, "location.latitude": lat, "location.longitude": lng, "languageCode": "ja"})
            if r.status_code == 200:
                d = r.json()
                wc = d.get("weatherCondition") or {}
                icon = wc.get("iconBaseUri")
                out = {"tz": (d.get("timeZone") or {}).get("id"), "temp": (d.get("temperature") or {}).get("degrees"),
                       "text": (wc.get("description") or {}).get("text"), "icon": f"{icon}.svg" if icon else None,
                       "day": d.get("isDaytime"), "source": "Google Weather"}
            else:
                log.warning(json.dumps({"event": "weather_error", "status": r.status_code, "body": r.text[:200]}, ensure_ascii=False))
        except Exception as e:
            log.warning("weather failed: %s", e)
        if not out.get("tz"):
            try:
                r = await http.get("https://api.open-meteo.com/v1/forecast", params={
                    "latitude": lat, "longitude": lng, "current": "temperature_2m,weather_code,is_day", "timezone": "auto"})
                if r.status_code == 200:
                    d = r.json(); c = d.get("current") or {}
                    out = {"tz": d.get("timezone"), "temp": c.get("temperature_2m"), "text": _WMO.get(c.get("weather_code")),
                           "icon": None, "day": bool(c.get("is_day")), "source": "Open-Meteo"}
            except Exception as e:
                log.warning("open-meteo failed: %s", e)
    if out.get("tz"):
        _local_cache[key] = (time.time(), out)
    return out


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
    lang = str(body.get("lang", "ja"))[:10]
    vname = str(body.get("voice", "") or "")[:20] or None
    if not text:
        return JSONResponse({"error": "empty"}, status_code=400)
    t0 = time.time()
    try:
        wav = await voice.synth(text, lang, vname)
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
