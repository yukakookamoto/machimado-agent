"""散歩している場所の「国のプロフィール」：国名・地域名・人口・政治体制・公用語・首都。
地域名は Google（Geocoding、だめなら Places の近くの場所）から、国の数字は Wikidata から取る（だめなら REST Countries）。
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


Q_POP = """SELECT ?pop ?date ?pref WHERE {
  ?c wdt:P297 "%s"; p:P1082 ?s. ?s ps:P1082 ?pop; wikibase:rank ?rk.
  FILTER(?rk != wikibase:DeprecatedRank) OPTIONAL { ?s pq:P585 ?date }
  BIND(IF(?rk = wikibase:PreferredRank, 1, 0) AS ?pref)
} ORDER BY DESC(?pref) DESC(?date) LIMIT 1"""
Q_LAB = """SELECT ?kind ?ja ?en WHERE {
  ?c wdt:P297 "%s".
  { ?c wdt:P122 ?item. BIND("gov" AS ?kind) } UNION { ?c wdt:P37 ?item. BIND("lang" AS ?kind) }
  UNION { ?c wdt:P36 ?item. BIND("cap" AS ?kind) } UNION { BIND(?c AS ?item) BIND("c" AS ?kind) }
  OPTIONAL { ?item rdfs:label ?ja FILTER(LANG(?ja) = "ja") }
  OPTIONAL { ?item rdfs:label ?en FILTER(LANG(?en) = "en") }
} LIMIT 60"""
ERRORS: list[str] = []


async def _sparql(http: httpx.AsyncClient, q: str) -> list[dict]:
    r = await http.get("https://query.wikidata.org/sparql", params={"query": q, "format": "json"},
                       headers={**UA, "Accept": "application/sparql-results+json"})
    if r.status_code != 200:
        raise RuntimeError(f"wikidata {r.status_code}: {r.text[:120]}")
    return r.json()["results"]["bindings"]


async def _wikidata(http: httpx.AsyncClient, cc: str) -> dict:
    val = lambda x, k: (x.get(k) or {}).get("value")
    pop = await _sparql(http, Q_POP % cc)
    labs = await _sparql(http, Q_LAB % cc)
    if not pop and not labs:
        return {}
    got: dict[str, list[tuple[str, str]]] = {}
    for x in labs:
        got.setdefault(val(x, "kind"), []).append((val(x, "ja") or "", val(x, "en") or ""))
    names = lambda k: list(dict.fromkeys((ja or en) for ja, en in got.get(k, []) if (ja or en)))
    p0 = pop[0] if pop else {}
    return {"population": int(float(val(p0, "pop"))) if val(p0, "pop") else None, "popYear": (val(p0, "date") or "")[:4],
            "government": names("gov")[:3], "languages": names("lang")[:4], "capital": (names("cap") or [""])[0],
            "countryName": (names("c") or [""])[0], "source": "Wikidata"}


async def _restcountries(http: httpx.AsyncClient, cc: str) -> dict:
    r = await http.get(f"https://restcountries.com/v3.1/alpha/{cc}", params={"fields": "population,languages,capital,translations"}, follow_redirects=True)
    r.raise_for_status()
    d = r.json(); d = d[0] if isinstance(d, list) else d
    return {"population": d.get("population"), "popYear": "", "government": [], "languages": list((d.get("languages") or {}).values())[:4],
            "capital": (d.get("capital") or [""])[0], "countryName": ((d.get("translations") or {}).get("jpn") or {}).get("common", ""),
            "source": "REST Countries"}


_ja_cache: dict[str, str] = {}
JP = __import__("re").compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


async def _to_japanese(words: list[str], gov_for: str = "") -> tuple[dict[str, str], str]:
    """英語や現地語のままの名前（州の名前・政治体制など）を、日本語の一般的な言い方にする（Gemini）。
    gov_for に国名があれば、その国の政治体制（日本語の短い言い方）も答えてもらう（データが取れなかったときの補い）"""
    todo = [w for w in dict.fromkeys(words) if w and not JP.search(w) and w not in _ja_cache]
    gov = _ja_cache.get("__gov__" + gov_for, "") if gov_for else ""
    if todo or (gov_for and not gov):
        import json
        from google.genai import types
        import agent
        prompt = ("次の語を、日本の地図帳や百科事典で使われる一般的な日本語の表記にしてください。"
                  "地名はカタカナや漢字の慣用表記（例：Catalunya→カタルーニャ州、England→イングランド、Taipei City→台北市）、"
                  "政治体制は日本語の用語（例：parliamentary monarchy→議会君主制、federal republic→連邦共和制）。\n"
                  "items には、元の語(src)と日本語(ja)の組を、元の語の数だけ入れる。\n語：" + json.dumps(todo, ensure_ascii=False)
                  + (f"\nあわせて government に、{gov_for}の政治体制を日本語の短い用語で1〜2個（例：立憲君主制・議院内閣制）。" if gov_for and not gov else ""))
        schema = {"type": "OBJECT", "properties": {"items": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "src": {"type": "STRING"}, "ja": {"type": "STRING"}}, "required": ["src", "ja"]}}, "government": {"type": "STRING"}}, "required": ["items"]}
        for _, client in agent.gemini_backends():
            done = False
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
                    g = str(d.get("government") or "").strip()
                    if gov_for and g and len(g) <= 40:
                        _ja_cache["__gov__" + gov_for] = gov = g
                    done = True
                    break
                except Exception as e:
                    ERRORS.append(f"ja {m}: {str(e)[:160]}"); del ERRORS[:-20]
            if done:
                break
    return {w: _ja_cache.get(w, w) for w in words}, gov


async def profile(lat: float, lng: float, key: str) -> dict:
    gk = f"{lat:.2f},{lng:.2f}"
    async with httpx.AsyncClient(timeout=12.0) as http:
        hit = _geo.get(gk)
        where = hit[1] if hit and time.time() - hit[0] < 86400 else await _where(http, lat, lng, key)
        if where:
            _geo[gk] = (time.time(), where)
        cc = (where.get("cc") or "").upper()
        if not cc:
            return {}
        hit = _facts.get(cc)
        facts = hit[1] if hit and time.time() - hit[0] < 7 * 86400 else None
        if facts is None:
            for fn in (_wikidata, _restcountries):
                try:
                    facts = await fn(http, cc)
                    if facts:
                        break
                except Exception as e:
                    log.warning("%s %s failed: %s", fn.__name__, cc, str(e)[:160])
                    ERRORS.append(f"{fn.__name__} {cc}: {str(e)[:160]}"); del ERRORS[:-20]
            if facts:
                _facts[cc] = (time.time(), facts)
            else:   # 取れなかったときも、しばらくは何度も問い合わせない
                _facts[cc] = (time.time() - 7 * 86400 + 600, {})
    facts = facts or {}
    out = {**where, "cc": cc, "flag": flag(cc), "country": where.get("country") or facts.get("countryName", ""),
           **{k: facts.get(k) for k in ("population", "popYear", "government", "languages", "capital", "source")}}
    words = [out.get("region") or "", out.get("city") or "", out.get("capital") or "", out.get("country") or ""] + list(out.get("government") or []) + list(out.get("languages") or [])
    try:
        ja, gov = await _to_japanese([w for w in words if w], "" if out.get("government") else (out.get("country") or cc))
        if gov and not out.get("government"):
            out["government"] = [x.strip() for x in gov.replace("、", "・").split("・") if x.strip()][:2]
        for k in ("region", "city", "capital", "country"):
            if out.get(k):
                out[k] = ja.get(out[k], out[k])
        for k in ("government", "languages"):
            if out.get(k):
                out[k] = list(dict.fromkeys(ja.get(x, x) for x in out[k]))
    except Exception as e:
        ERRORS.append(f"ja: {str(e)[:160]}")
    return out
