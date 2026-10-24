"""散歩している場所の「国のプロフィール」：国名・地域名・人口・政治体制・公用語・首都。
地域名は Google（Geocoding、だめなら Places の近くの場所）から、国の数字は Wikidata から取る（だめなら REST Countries）。
"""
from __future__ import annotations

import logging
import time

import httpx

log = logging.getLogger("machimado.country")
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


SPARQL = """SELECT ?pop ?popDate ?rk ?govLabel ?langLabel ?capLabel ?cLabel WHERE {
  ?c wdt:P297 "%s".
  OPTIONAL { ?c p:P1082 ?ps. ?ps ps:P1082 ?pop; wikibase:rank ?rk. OPTIONAL { ?ps pq:P585 ?popDate } }
  OPTIONAL { ?c wdt:P122 ?gov }
  OPTIONAL { ?c wdt:P37 ?lang }
  OPTIONAL { ?c wdt:P36 ?cap }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "ja,en". }
}"""


async def _wikidata(http: httpx.AsyncClient, cc: str) -> dict:
    r = await http.get("https://query.wikidata.org/sparql", params={"query": SPARQL % cc, "format": "json"},
                       headers={**UA, "Accept": "application/sparql-results+json"})
    r.raise_for_status()
    rows = r.json()["results"]["bindings"]
    if not rows:
        return {}
    val = lambda x, k: (x.get(k) or {}).get("value")
    pops = [((val(x, "rk") or "").endswith("PreferredRank"), val(x, "popDate") or "", val(x, "pop")) for x in rows
            if val(x, "pop") and not (val(x, "rk") or "").endswith("DeprecatedRank")]
    best = max(pops) if pops else None
    uniq = lambda k: list(dict.fromkeys(v for v in (val(x, k) for x in rows) if v and not v.startswith("http")))
    return {"population": int(float(best[2])) if best else None, "popYear": (best[1] or "")[:4] if best else "",
            "government": uniq("govLabel")[:3], "languages": uniq("langLabel")[:4], "capital": (uniq("capLabel") or [""])[0],
            "countryName": (uniq("cLabel") or [""])[0], "source": "Wikidata"}


async def _restcountries(http: httpx.AsyncClient, cc: str) -> dict:
    r = await http.get(f"https://restcountries.com/v3.1/alpha/{cc}", params={"fields": "population,languages,capital,translations"})
    r.raise_for_status()
    d = r.json(); d = d[0] if isinstance(d, list) else d
    return {"population": d.get("population"), "popYear": "", "government": [], "languages": list((d.get("languages") or {}).values())[:4],
            "capital": (d.get("capital") or [""])[0], "countryName": ((d.get("translations") or {}).get("jpn") or {}).get("common", ""),
            "source": "REST Countries"}


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
            if facts:
                _facts[cc] = (time.time(), facts)
    facts = facts or {}
    return {**where, "cc": cc, "flag": flag(cc), "country": where.get("country") or facts.get("countryName", ""),
            **{k: facts.get(k) for k in ("population", "popYear", "government", "languages", "capital", "source")}}
