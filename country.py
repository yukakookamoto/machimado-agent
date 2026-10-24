"""散歩している場所の「国のプロフィール」：国名・地域名・人口・政治体制・公用語・首都。
地域名は Google（Geocoding、だめなら Places の近くの場所）から、人口は世界銀行のデータ（最新年）、
首都・公用語・政治体制と、地名の日本語表記は Gemini に答えてもらう（変わりにくい基本情報だけ）。
"""
from __future__ import annotations

import logging
import time

import httpx

log = logging.getLogger("machimado.country")
try:   # Wikimedia は連絡先入りの User-Agent を求める（ないと 429 になりやすい）
    from agent import UA
except Exception:
    UA = {"User-Agent": "machimado-agent/1.0 (hackathon demo)"}
_geo: dict[str, tuple[float, dict]] = {}
_facts: dict[str, tuple[float, dict]] = {}


def flag(cc: str) -> str:
    cc = (cc or "").upper()
    return "".join(chr(0x1F1E6 + ord(c) - 65) for c in cc) if len(cc) == 2 and cc.isalpha() else ""


def _pick(comps: list[dict], kind: str, long_key: str, short_key: str) -> tuple[str, str]:
    for c in comps:
        if kind in (c.get("types") or []):
            return c.get(long_key) or "", c.get(short_key) or ""
    return "", ""


async def _where(http: httpx.AsyncClient, lat: float, lng: float, key: str) -> dict:
    """座標 → 国コード・国名・州や県・市（日本語）"""
    try:
        r = await http.get("https://maps.googleapis.com/maps/api/geocode/json",
                           params={"latlng": f"{lat},{lng}", "language": "ja", "key": key})
        j = r.json()
        if j.get("status") == "OK":
            comps = [c for res in j.get("results", [])[:3] for c in res.get("address_components", [])]
            country, cc = _pick(comps, "country", "long_name", "short_name")
            region, _ = _pick(comps, "administrative_area_level_1", "long_name", "short_name")
            city, _ = _pick(comps, "locality", "long_name", "short_name")
            if cc:
                return {"cc": cc, "country": country, "region": region, "city": city}
        else:
            log.warning("geocode %s: %s", j.get("status"), str(j.get("error_message", ""))[:160])
    except Exception as e:
        log.warning("geocode failed: %s", e)
    try:   # Geocoding が使えないときは、Places（New）でいちばん近い場所の住所から
        r = await http.post("https://places.googleapis.com/v1/places:searchNearby", json={
            "maxResultCount": 1, "languageCode": "ja",
            "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lng}, "radius": 500.0}}},
            headers={"X-Goog-Api-Key": key, "X-Goog-FieldMask": "places.addressComponents"})
        comps = ((r.json().get("places") or [{}])[0]).get("addressComponents") or []
        country, cc = _pick(comps, "country", "longText", "shortText")
        region, _ = _pick(comps, "administrative_area_level_1", "longText", "shortText")
        city, _ = _pick(comps, "locality", "longText", "shortText")
        if cc:
            return {"cc": cc, "country": country, "region": region, "city": city}
    except Exception as e:
        log.warning("places nearby failed: %s", e)
    return {}


ERRORS: list[str] = []
_ja_cache: dict[str, str] = {}
_basics: dict[str, dict] = {}
JP = __import__("re").compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


def _err(msg: str) -> None:
    ERRORS.append(msg[:200]); del ERRORS[:-20]
    log.warning(msg[:200])


async def _population(http: httpx.AsyncClient, cc: str) -> dict:
    """世界銀行の人口（いちばん新しい年）。台湾など載っていない国・地域は空"""
    r = await http.get(f"https://api.worldbank.org/v2/country/{cc}/indicator/SP.POP.TOTL",
                       params={"format": "json", "mrnev": 1}, headers=UA, follow_redirects=True)
    d = r.json()
    rows = d[1] if isinstance(d, list) and len(d) > 1 and d[1] else []
    for x in rows:
        if x.get("value"):
            return {"population": int(x["value"]), "popYear": str(x.get("date") or ""), "popSource": "世界銀行"}
    return {}


async def _gemini(cc: str, country: str, words: list[str]) -> dict:
    """首都・公用語・政治体制（日本語）と、地名の日本語表記。人口が取れない国は、その目安も"""
    import json
    from google.genai import types
    import agent
    todo = [w for w in dict.fromkeys(words) if w and not JP.search(w) and w not in _ja_cache]
    need_basics = cc not in _basics
    if not todo and not need_basics:
        return _basics.get(cc, {})
    prompt = (f"国・地域コード {cc}（{country}）について、日本の百科事典や地図帳で使われる日本語で答えてください。\n"
              "capital：首都。languages：公用語（多いときはおもなもの3つまで）。government：政治体制を短い用語で1〜2個（例：立憲君主制、議院内閣制、連邦共和制、大統領制）。\n"
              "population_estimate：人口のおおよその数（整数）。わからなければ0。\n"
              "items：次の語の日本語の慣用表記（地名はカタカナ・漢字。例：Catalunya→カタルーニャ州、England→イングランド、Taipei City→台北市）。"
              "元の語(src)と日本語(ja)の組を、語の数だけ。\n語：" + json.dumps(todo, ensure_ascii=False))
    schema = {"type": "OBJECT", "properties": {
        "capital": {"type": "STRING"}, "languages": {"type": "ARRAY", "items": {"type": "STRING"}},
        "government": {"type": "ARRAY", "items": {"type": "STRING"}}, "population_estimate": {"type": "INTEGER"},
        "items": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"src": {"type": "STRING"}, "ja": {"type": "STRING"}}, "required": ["src", "ja"]}}},
        "required": ["capital", "languages", "government", "items"]}
    for _, client in agent.gemini_backends():
        for m in agent.VISION_MODELS or ["gemini-2.5-flash"]:
            try:
                kw = {"response_mime_type": "application/json", "response_schema": schema, "temperature": 0}
                if m.startswith("gemini-2.5"):
                    kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                resp = await client.aio.models.generate_content(model=m, contents=prompt, config=types.GenerateContentConfig(**kw))
                d = json.loads(resp.text or "{}")
                for it in d.get("items") or []:
                    src, ja = str(it.get("src") or ""), str(it.get("ja") or "").strip()
                    if src in todo and ja and len(ja) <= 40:
                        _ja_cache[src] = ja
                if need_basics:
                    clean = lambda xs: [str(x).strip() for x in (xs or []) if str(x).strip() and len(str(x)) <= 30]
                    _basics[cc] = {"capital": str(d.get("capital") or "")[:30], "languages": clean(d.get("languages"))[:3],
                                   "government": clean(d.get("government"))[:2], "popEstimate": int(d.get("population_estimate") or 0)}
                return _basics.get(cc, {})
            except Exception as e:
                _err(f"gemini {m}: {e}")
    return _basics.get(cc, {})


async def profile(lat: float, lng: float, key: str) -> dict:
    gk = f"{lat:.2f},{lng:.2f}"
    async with httpx.AsyncClient(timeout=10.0) as http:
        hit = _geo.get(gk)
        where = hit[1] if hit and time.time() - hit[0] < 86400 else await _where(http, lat, lng, key)
        if where:
            _geo[gk] = (time.time(), where)
        cc = (where.get("cc") or "").upper()
        if not cc:
            return {}
        hit = _facts.get(cc)
        pop = hit[1] if hit and time.time() - hit[0] < 7 * 86400 else None
        if pop is None:
            try:
                pop = await _population(http, cc)
            except Exception as e:
                _err(f"worldbank {cc}: {type(e).__name__} {e}"); pop = {}
            _facts[cc] = (time.time() if pop else time.time() - 7 * 86400 + 3600, pop)
    out = {**where, "cc": cc, "flag": flag(cc), **pop}
    b = await _gemini(cc, where.get("country") or cc, [where.get("region") or "", where.get("city") or ""])
    for k in ("region", "city"):
        if out.get(k):
            out[k] = _ja_cache.get(out[k], out[k])
    out.update({k: b.get(k) for k in ("capital", "languages", "government")})
    if not out.get("population") and b.get("popEstimate"):
        out.update({"population": b["popEstimate"], "popYear": "", "popSource": "推定"})
    out["source"] = "人口：" + (out.get("popSource") or "—") + "／首都・公用語・政治体制：Gemini"
    return out
