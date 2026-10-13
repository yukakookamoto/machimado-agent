"""もちこの声：Gemini の音声生成（TTS）で、案内の文を声にする。
返ってくるのは 24kHz・16bit・モノラルの生の音（PCM）なので、WAV の頭を付けてブラウザで鳴らせるようにする。
同じ文はくり返し作らない（メモリにしまっておく）。
"""
from __future__ import annotations

import hashlib
import logging
import os
import struct
from collections import OrderedDict

from google import genai
from google.genai import types

log = logging.getLogger("machimado.voice")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# 使えるモデルから順に試す。どれも無ければ、名前に tts が入るモデルを一覧から探す
TTS_MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_TTS_MODELS", "gemini-2.5-flash-preview-tts,gemini-2.5-flash-tts,gemini-2.5-pro-preview-tts").split(",") if m.strip()]
VOICE_NAME = os.environ.get("MOCHIKO_VOICE", "Leda")   # 若くやわらかい声
STYLE = os.environ.get("MOCHIKO_STYLE",
                       "小さくてかわいい案内役の子が、となりを歩きながら、明るく、やさしく、少しゆっくり話すように読んでください")

_cache: "OrderedDict[str, bytes]" = OrderedDict()
_CACHE_MAX = 300
_models: list[str] = list(TTS_MODELS)
_discovered = False


def _wav(pcm: bytes, rate: int = 24000) -> bytes:
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm)) + pcm)


def _rate_of(mime: str) -> int:
    for part in (mime or "").split(";"):
        part = part.strip()
        if part.startswith("rate="):
            try:
                return int(part[5:])
            except ValueError:
                pass
    return 24000


async def _discover(client: genai.Client) -> None:
    """設定したモデルが無いとき、使える音声モデルを一覧から探す（自分で直る）"""
    global _discovered
    _discovered = True
    try:
        found = []
        async for m in await client.aio.models.list():
            name = (m.name or "").split("/")[-1]
            if "tts" in name and name not in _models:
                found.append(name)
        found.sort(reverse=True)   # 新しそうなものから
        _models.extend(found)
        log.info("tts models discovered: %s", found)
    except Exception as e:
        log.warning("tts model list failed: %s", e)


# ---- ふだんの声：Google Cloud Text-to-Speech（Chirp 3 HD）。日本語の抑揚がいちばん自然 ----
MAPS_API_KEY = os.environ.get("MAPS_API_KEY", "")   # この鍵で Cloud Text-to-Speech API も使えるようにしてある
CHIRP_VOICE = os.environ.get("MOCHIKO_CHIRP", "Aoede")   # Chirp 3 HD の声の名前（Aoede / Leda / Kore / Zephyr など）
CHIRP_NAMES = {"Aoede", "Leda", "Kore", "Zephyr", "Callirrhoe", "Despina", "Erinome", "Laomedeia", "Pulcherrima", "Vindemiatrix",
               "Achernar", "Autonoe", "Gacrux", "Sulafat", "Puck", "Charon", "Fenrir", "Orus"}
LOCALES = {"ja": "ja-JP", "fr": "fr-FR", "en": "en-US", "it": "it-IT", "es": "es-ES", "de": "de-DE", "zh": "cmn-CN", "ko": "ko-KR", "pt": "pt-BR", "ru": "ru-RU"}


async def _chirp(text: str, locale: str, voice: str) -> bytes:
    import base64
    import httpx
    body = {"input": {"text": text}, "voice": {"languageCode": locale, "name": f"{locale}-Chirp3-HD-{voice}"},
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000}}
    async with httpx.AsyncClient(timeout=20.0) as http:
        r = await http.post("https://texttospeech.googleapis.com/v1/text:synthesize", params={"key": MAPS_API_KEY}, json=body)
    if r.status_code != 200:
        raise RuntimeError(f"cloud tts {r.status_code}: {r.text[:200]}")
    data = base64.b64decode(r.json()["audioContent"])
    return data if data[:4] == b"RIFF" else _wav(data, 24000)


def _put(key: str, wav: bytes) -> bytes:
    _cache[key] = wav
    if len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return wav


async def synth(text: str, lang_hint: str = "ja", voice: str | None = None) -> bytes:
    """文を声（WAV）にする。Chirp 3 HD → だめなら Gemini の音声生成。両方だめなら例外（画面側はブラウザの声）"""
    text = text.strip()[:700]
    voice = voice if voice in CHIRP_NAMES else CHIRP_VOICE
    lang = (lang_hint or "ja").split("-")[0].lower()
    key = hashlib.sha1(f"{voice}|{lang}|{text}".encode()).hexdigest()
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    if MAPS_API_KEY:
        try:
            return _put(key, await _chirp(text, LOCALES.get(lang, "en-US"), voice))
        except Exception as e:
            log.warning("chirp tts failed (%s): %s", lang, str(e)[:200])
    lang_hint = "ja" if lang == "ja" else "other"
    if not GEMINI_API_KEY and not os.environ.get("VERTEX_PROJECT"):
        raise RuntimeError("GEMINI_API_KEY がありません")
    vp = os.environ.get("VERTEX_PROJECT", "")
    client = (genai.Client(vertexai=True, project=vp, location=os.environ.get("VERTEX_LOCATION", "global"))
              if vp else genai.Client(api_key=GEMINI_API_KEY))
    style = STYLE if lang_hint == "ja" else "旅先で現地の人に話しかけるように、はっきり、ゆっくり読んでください"
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE_NAME))))
    last = None
    i = 0
    while i < len(_models):
        model = _models[i]
        try:
            resp = await client.aio.models.generate_content(model=model, contents=f"{style}：\n{text}", config=cfg)
            part = resp.candidates[0].content.parts[0]
            data = part.inline_data.data
            wav = _wav(data, _rate_of(part.inline_data.mime_type))
            if i:   # うまくいったモデルを先頭へ
                _models.insert(0, _models.pop(i))
            _cache[key] = wav
            if len(_cache) > _CACHE_MAX:
                _cache.popitem(last=False)
            return wav
        except Exception as e:
            msg = str(e)
            last = e
            if any(x in msg for x in ("404", "NOT_FOUND", "not found", "not supported")):
                log.warning("tts model %s unavailable: %s", model, msg[:160])
                _models.pop(i)   # 次からは試さない
                if i >= len(_models) and not _discovered:
                    await _discover(client)
                continue
            raise
    raise RuntimeError(f"使える音声モデルがありません（{str(last)[:160]}）")
