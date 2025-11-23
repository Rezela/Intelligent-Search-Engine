import requests
import googlemaps
import re
import yfinance as yf
from datetime import datetime, timedelta

import os
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 定义统一接口
class BaseAPIHandler:
    def handle(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError





# 天气 API
class HKOOpenDataClient:
    BASE_URL = "https://data.weather.gov.hk/weatherAPI/opendata/weather.php"
    TC_TRACK_URL = "https://data.weather.gov.hk/weatherAPI/opendata/tcTrack"

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()

    def fetch(self, data_type: str, lang: str = "tc", extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        params = {"dataType": data_type, "lang": lang}
        if extra:
            params.update(extra)
        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def warning_summary(self, lang: str = "tc") -> Optional[Dict[str, Any]]:
        return self.fetch("warnsum", lang=lang)

    def special_weather_tips(self, lang: str = "tc") -> Optional[Dict[str, Any]]:
        return self.fetch("swt", lang=lang)

    def uv_index(self, lang: str = "tc") -> Optional[Dict[str, Any]]:
        return self.fetch("uvindex", lang=lang)

    def local_weather_forecast(self, lang: str = "tc") -> Optional[Dict[str, Any]]:
        return self.fetch("flw", lang=lang)

    def tc_list(self, lang: str = "tc") -> Optional[Dict[str, Any]]:
        params = {"lang": lang}
        try:
            resp = self.session.get(self.TC_TRACK_URL, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def tc_track(self, storm_id: str, lang: str = "tc") -> Optional[Dict[str, Any]]:
        if not storm_id:
            return None
        params = {"stormId": storm_id, "lang": lang}
        try:
            resp = self.session.get(self.TC_TRACK_URL, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None


class WeatherAPIHandler(BaseAPIHandler):
    GEO_URL = "https://api.openweathermap.org/geo/1.0/direct"
    CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
    FORECAST_HOURLY_URL = "https://api.openweathermap.org/data/2.5/forecast"
    FORECAST_DAILY_URL = "https://api.openweathermap.org/data/2.5/forecast/daily"
    AIR_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
    AIR_FORECAST_URL = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"
    FALLBACK_CITIES = [
        "北京",
        "上海",
        "广州",
        "深圳",
        "香港",
        "澳门",
        "巴黎",
        "伦敦",
        "纽约",
        "Beijing",
        "Shanghai",
        "Shenzhen",
        "Hong Kong",
        "Macau",
        "Guangzhou",
        "London",
        "New York",
        "Singapore",
    ]
    HK_KEYWORDS = [
        "hong kong",
        "香港",
        "香江",
        "天文台",
        "天文臺",
        "hko",
        "八號",
        "八号",
        "no.8",
        "八號風球",
        "八號風暴信號",
        "typhoon signal",
        "tropical cyclone",
        "熱帶氣旋",
        "台風",
        "暴雨警告",
        "black rain",
        "amber rain",
    ]

    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")
        self.session = requests.Session()
        self.lang = os.getenv("WEATHER_LANG", "zh_cn")
        self.units = os.getenv("WEATHER_UNITS", "metric")
        self._geo_cache: Dict[str, Dict[str, Any]] = {}
        self.hko_client = HKOOpenDataClient(session=self.session)

    def handle(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key:
            return {"error": "未配置 WEATHER_API_KEY"}

        if self._is_library_signal_query(query):
            return self._library_signal_policy_response()

        city = self._resolve_city(query, metadata)
        if not city:
            return {"error": "未能识别城市，请输入类似“北京天气”的查询。"}

        location = self._geocode_city(city)
        if not location:
            return {"error": f"无法解析 {city} 的地理位置"}

        lat, lon = location["lat"], location["lon"]
        current_raw = self._fetch_current_weather(lat, lon)
        if not current_raw or "main" not in current_raw:
            return {"error": f"无法获取 {location['name']} 的天气信息"}

        tz_offset = current_raw.get("timezone", 0)
        hourly_raw = self._fetch_hourly_forecast(lat, lon)
        daily_raw = self._fetch_daily_forecast(lat, lon)
        air_raw = self._fetch_air_quality(lat, lon)
        air_forecast_raw = self._fetch_air_quality_forecast(lat, lon)

        current_detail = self._extract_current(current_raw, tz_offset)
        hourly_detail = self._extract_hourly_forecast(hourly_raw, tz_offset)
        daily_detail = self._extract_daily_forecast(daily_raw, tz_offset)
        air_detail = self._extract_air_quality(air_raw)
        air_forecast_detail = self._extract_air_forecast(air_forecast_raw)
        advisories = self._deduce_advisories(current_detail, hourly_detail, air_detail)
        hko_context = self._fetch_hko_context(query, city)

        summary = self._build_summary(
            location["name"],
            current_detail,
            hourly_detail,
            daily_detail,
            air_detail,
            advisories,
            self._infer_time_scope(query, metadata),
            hko_context,
        )

        return {
            "summary": summary,
            "location": {
                "name": location["name"],
                "country": location.get("country"),
                "lat": lat,
                "lon": lon,
            },
            "current": current_detail,
            "forecast_hourly": hourly_detail,
            "forecast_daily": daily_detail,
            "air_quality": {"current": air_detail, "forecast": air_forecast_detail},
            "advisories": advisories,
            "typhoon_signal": hko_context.get("typhoon_signal"),
            "warnings": hko_context.get("warnings"),
            "special_weather_tips": hko_context.get("special_weather_tips"),
            "uv_index": hko_context.get("uv_index"),
            "raw": {
                "current": current_raw,
                "forecast_hourly": hourly_raw,
                "forecast_daily": daily_raw,
                "air_quality": air_raw,
                "air_quality_forecast": air_forecast_raw,
                "hko": hko_context.get("raw"),
            },
        }

    def _request(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        params["appid"] = self.api_key
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def _resolve_city(self, query: str, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if metadata:
            entities = metadata.get("entities") or {}
            entity_city = entities.get("location")
            if isinstance(entity_city, str) and entity_city.strip():
                return entity_city.strip()

        lowered = query.lower()
        for city in self.FALLBACK_CITIES:
            if city.lower() in lowered:
                return city
        if any(keyword in lowered for keyword in self.HK_KEYWORDS):
            return "Hong Kong"
        return None

    def _geocode_city(self, city: str) -> Optional[Dict[str, Any]]:
        cache_key = city.lower()
        if cache_key in self._geo_cache:
            return self._geo_cache[cache_key]
        params = {"q": city, "limit": 1, "appid": self.api_key}
        data = self._request(self.GEO_URL, params)
        if data:
            record = data[0]
            normalized = {
                "name": record.get("name", city),
                "country": record.get("country"),
                "lat": record.get("lat"),
                "lon": record.get("lon"),
            }
            self._geo_cache[cache_key] = normalized
            return normalized
        return None

    def _fetch_current_weather(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        params = {"lat": lat, "lon": lon, "units": self.units, "lang": self.lang}
        return self._request(self.CURRENT_URL, params)

    def _fetch_hourly_forecast(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        params = {"lat": lat, "lon": lon, "cnt": 16, "units": self.units, "lang": self.lang}
        return self._request(self.FORECAST_HOURLY_URL, params)

    def _fetch_daily_forecast(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        params = {"lat": lat, "lon": lon, "cnt": 7, "units": self.units, "lang": self.lang}
        return self._request(self.FORECAST_DAILY_URL, params)

    def _fetch_air_quality(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        params = {"lat": lat, "lon": lon}
        return self._request(self.AIR_URL, params)

    def _fetch_air_quality_forecast(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        params = {"lat": lat, "lon": lon}
        return self._request(self.AIR_FORECAST_URL, params)

    def _is_library_signal_query(self, query: str) -> bool:
        lowered = query.lower()
        return (
            ("公共圖書館" in query or "公共图书馆" in query or "public library" in lowered)
            and ("signal" in lowered or "信號" in query or "信号" in query or "風球" in query)
        )

    def _library_signal_policy_response(self) -> Dict[str, Any]:
        summary = (
            "香港公共圖書館會在天文台發出八號或以上熱帶氣旋警告信號、或黑色暴雨警告時暫停開放；"
            "當紅色/黃色暴雨或三號風球生效時通常維持有限度服務，視乎最新官方公告。"
        )
        return {
            "summary": summary,
            "policy": {
                "closure_signals": ["Typhoon Signal No.8 或以上", "黑色暴雨警告"],
                "limited_service": ["Typhoon Signal No.3", "紅色/黃色暴雨警告"],
            },
        }

    def _fetch_hko_context(self, query: str, city: str) -> Dict[str, Any]:
        if not city:
            return {}
        if city.lower() not in {"hong kong", "hong kong island", "香港"} and not self._looks_like_hko_request(query):
            return {}
        lang = "tc" if self.lang in ("zh_tw", "tc") else ("sc" if self.lang in ("zh_cn", "sc") else "en")
        warnings_raw = self.hko_client.warning_summary(lang=lang)
        swt_raw = self.hko_client.special_weather_tips(lang=lang)
        uv_raw = self.hko_client.uv_index(lang=lang)
        tc_list = self.hko_client.tc_list(lang=lang)
        storm_id = self._latest_storm_id(tc_list)
        track_raw = self.hko_client.tc_track(storm_id, lang=lang) if storm_id else None
        track_summary = self._parse_tc_track(track_raw)

        typhoon_signal, warnings = self._parse_warning_summary(warnings_raw, storm_id, track_summary)
        uv_info = self._parse_uv_index(uv_raw)
        tips = self._parse_special_weather_tips(swt_raw)
        if not warnings:
            warnings = ["香港天文台目前沒有生效的天氣警告。"]
        if not typhoon_signal:
            typhoon_signal = {
                "code": "none",
                "message": "天文台目前沒有懸掛任何熱帶氣旋警告信號。",
                "issueTime": (warnings_raw or {}).get("updateTime"),
                "track_summary": track_summary,
            }
        return {
            "typhoon_signal": typhoon_signal,
            "warnings": warnings,
            "special_weather_tips": tips,
            "uv_index": uv_info,
            "tc_track": track_summary,
            "raw": {
                "warning_summary": warnings_raw,
                "special_weather_tips": swt_raw,
                "uv_index": uv_raw,
                "tc_list": tc_list,
                "tc_track": track_raw,
            },
        }

    def _looks_like_hko_request(self, query: str) -> bool:
        lowered = query.lower()
        return any(keyword in lowered for keyword in self.HK_KEYWORDS)

    def _latest_storm_id(self, payload: Optional[Dict[str, Any]]) -> Optional[str]:
        if not payload:
            return None
        storms = payload.get("tcList") or payload.get("data") or payload.get("storms") or []
        if not storms:
            return None
        def _order_key(item):
            return item.get("issueTime") or item.get("updateTime") or item.get("time") or ""
        storms_sorted = sorted(storms, key=_order_key, reverse=True)
        primary = storms_sorted[0]
        for key in ("stormId", "id", "storm_id"):
            if primary.get(key):
                return primary[key]
        return None

    def _parse_tc_track(self, payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not payload:
            return None
        track_points = payload.get("tcTrack") or payload.get("track") or payload.get("points") or []
        if not track_points:
            return None
        latest_point = track_points[-1]
        storm_name = (
            payload.get("name") or payload.get("stormName") or payload.get("tcName") or ""
        )
        return {
            "stormName": storm_name,
            "stormId": payload.get("stormId"),
            "latest": {
                "time": latest_point.get("time") or latest_point.get("recordTime"),
                "lat": latest_point.get("lat"),
                "lon": latest_point.get("lng") or latest_point.get("lon"),
                "intensity": latest_point.get("intensity"),
                "category": latest_point.get("cat") or latest_point.get("category"),
                "distanceToHK": latest_point.get("distanceToHK"),
            },
            "points": track_points,
        }

    def _parse_warning_summary(self, payload: Optional[Dict[str, Any]], storm_id: Optional[str], track_summary: Optional[Dict[str, Any]]) -> (Optional[Dict[str, Any]], List[str]):
        if not payload:
            return None, []
        details = payload.get("details") or []
        warnings: List[str] = []
        typhoon_signal = None
        for item in details:
            message = item.get("warningMessage") or item.get("name")
            if not message:
                continue
            warnings.append(message)
            warning_type = (item.get("warningType") or "").lower()
            code = item.get("warningStatementCode") or item.get("warningCode")
            issue_time = item.get("issueTime") or payload.get("updateTime")
            if warning_type == "tc":
                typhoon_signal = {
                    "code": code,
                    "message": message,
                    "issueTime": issue_time,
                    "track_summary": track_summary,
                    "stormId": storm_id,
                }
        return typhoon_signal, warnings

    def _parse_uv_index(self, payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not payload:
            return None
        records = payload.get("uvindex") or payload.get("data") or []
        if not records:
            return None
        record = records[0]
        try:
            value = float(record.get("value"))
        except (TypeError, ValueError):
            value = record.get("value")
        return {
            "value": value,
            "desc": record.get("desc") or record.get("description"),
            "place": record.get("place"),
            "recordTime": record.get("recordTime") or payload.get("updateTime"),
        }

    def _parse_special_weather_tips(self, payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not payload:
            return []
        tips = payload.get("specialWeatherTips") or payload.get("data") or []
        formatted: List[Dict[str, Any]] = []
        for tip in tips:
            if not tip:
                continue
            formatted.append(
                {
                    "title": tip.get("title") or "",
                    "content": tip.get("content") or tip.get("desc") or "",
                    "issueTime": tip.get("issueTime") or tip.get("updateTime"),
                }
            )
        return formatted

    def _extract_current(self, payload: Dict[str, Any], tz_offset: int) -> Dict[str, Any]:
        weather = (payload.get("weather") or [{}])[0]
        wind = payload.get("wind") or {}
        main = payload.get("main") or {}
        sunrise = (payload.get("sys") or {}).get("sunrise")
        sunset = (payload.get("sys") or {}).get("sunset")
        return {
            "description": weather.get("description"),
            "temperature": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "pressure": main.get("pressure"),
            "wind_speed": wind.get("speed"),
            "wind_direction": wind.get("deg"),
            "visibility": payload.get("visibility"),
            "sunrise": sunrise,
            "sunrise_local": self._format_time(sunrise, tz_offset),
            "sunset": sunset,
            "sunset_local": self._format_time(sunset, tz_offset),
            "timestamp": payload.get("dt"),
            "timestamp_local": self._format_time(payload.get("dt"), tz_offset, "%Y-%m-%d %H:%M"),
        }

    def _extract_hourly_forecast(self, payload: Optional[Dict[str, Any]], tz_offset: int) -> List[Dict[str, Any]]:
        if not payload:
            return []
        entries = payload.get("list") or []
        highlights: List[Dict[str, Any]] = []
        for item in entries[:16]:
            weather = (item.get("weather") or [{}])[0]
            main = item.get("main") or {}
            highlights.append(
                {
                    "time": item.get("dt"),
                    "time_local": self._format_time(item.get("dt"), tz_offset, "%m-%d %H:%M"),
                    "temperature": main.get("temp"),
                    "feels_like": main.get("feels_like"),
                    "description": weather.get("description"),
                    "humidity": main.get("humidity"),
                    "wind_speed": (item.get("wind") or {}).get("speed"),
                    "precip_probability": item.get("pop"),
                }
            )
        return highlights

    def _extract_daily_forecast(self, payload: Optional[Dict[str, Any]], tz_offset: int) -> List[Dict[str, Any]]:
        if not payload:
            return []
        entries = payload.get("list") or []
        days: List[Dict[str, Any]] = []
        for item in entries[:5]:
            weather = (item.get("weather") or [{}])[0]
            temps = item.get("temp") or {}
            days.append(
                {
                    "date": self._format_time(item.get("dt"), tz_offset, "%Y-%m-%d"),
                    "temp_min": temps.get("min"),
                    "temp_max": temps.get("max"),
                    "temp_day": temps.get("day"),
                    "temp_night": temps.get("night"),
                    "description": weather.get("description"),
                    "wind_speed": item.get("speed"),
                    "humidity": item.get("humidity"),
                    "pop": item.get("pop"),
                }
            )
        return days

    def _extract_air_quality(self, payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not payload:
            return None
        records = payload.get("list") or []
        if not records:
            return None
        main = records[0].get("main") or {}
        components = records[0].get("components") or {}
        return {
            "aqi": main.get("aqi"),
            "category": self._map_aqi(main.get("aqi")),
            "components": components,
            "timestamp": records[0].get("dt"),
        }

    def _extract_air_forecast(self, payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not payload:
            return None
        entries = payload.get("list") or []
        for item in entries:
            main = item.get("main") or {}
            if main.get("aqi"):
                return {
                    "aqi": main.get("aqi"),
                    "category": self._map_aqi(main.get("aqi")),
                    "timestamp": item.get("dt"),
                    "components": item.get("components"),
                }
        return None

    def _build_summary(
        self,
        city: str,
        current: Dict[str, Any],
        hours: List[Dict[str, Any]],
        days: List[Dict[str, Any]],
        air_quality: Optional[Dict[str, Any]],
        advisories: List[str],
        time_scope: Optional[str],
        hko_context: Dict[str, Any],
    ) -> str:
        parts = [
            f"{city}当前{current.get('description') or '天气情况'}，"
            f"气温 {self._format_value(current.get('temperature'), '°C')}，"
            f"体感 {self._format_value(current.get('feels_like'), '°C')}，"
            f"湿度 {self._format_value(current.get('humidity'), '%')}，"
            f"风速 {self._format_value(current.get('wind_speed'), 'm/s')}。"
        ]
        if current.get("sunrise_local") and current.get("sunset_local"):
            parts.append(f"日出 {current['sunrise_local']}，日落 {current['sunset_local']}。")

        slot = self._select_hourly_slot(hours, time_scope)
        if slot:
            parts.append(
                f"{self._describe_period(time_scope)}（{slot.get('time_local')}）"
                f"{slot.get('description') or ''}，温度 {self._format_value(slot.get('temperature'), '°C')}"
                + (f"，降水概率 {int((slot.get('precip_probability') or 0) * 100)}%" if slot.get("precip_probability") is not None else "")
            )

        if days:
            tomorrow = days[0]
            parts.append(
                f"明日 {tomorrow['date']} 预计 {tomorrow.get('description', '')}，"
                f"温度 {self._format_value(tomorrow.get('temp_min'), '°C')}~{self._format_value(tomorrow.get('temp_max'), '°C')}。"
            )

        if air_quality:
            parts.append(f"空气质量 {air_quality.get('category')}（AQI {air_quality.get('aqi')}）。")

        if advisories:
            parts.append("提示：" + "；".join(advisories) + "。")

        ty_signal = hko_context.get("typhoon_signal")
        if ty_signal:
            parts.append(f"热带气旋警告：{ty_signal.get('message')}")

        warnings = hko_context.get("warnings") or []
        if warnings:
            parts.append("香港天气警告：" + "；".join(warnings[:2]))

        uv_index = hko_context.get("uv_index")
        if uv_index:
            parts.append(
                f"15分钟紫外线指数 {uv_index.get('value')}（{uv_index.get('desc') or '—'}）"
            )

        tc_track = hko_context.get("tc_track")
        if tc_track:
            latest = tc_track.get("latest") or {}
            storm = tc_track.get("stormName") or tc_track.get("stormId") or "热带气旋"
            lat = latest.get("lat")
            lon = latest.get("lon")
            time = latest.get("time")
            intensity = latest.get("intensity") or latest.get("category")
            parts.append(
                f"{storm} 最新位置 ({lat}, {lon}) 于 {time}，强度 {intensity or '未知'}。"
            )

        tips = hko_context.get("special_weather_tips") or []
        if tips:
            tip = tips[0]
            content = tip.get("content") or tip.get("title")
            if content:
                parts.append(f"特别天气提示：{content}")

        return "".join(parts)

    def _format_value(self, value: Optional[float], unit: str) -> str:
        if value is None:
            return "—"
        try:
            return f"{round(float(value), 1)}{unit}"
        except Exception:
            return f"{value}{unit}"

    def _format_time(self, ts: Optional[int], tz_offset: int, fmt: str = "%H:%M") -> Optional[str]:
        if ts is None:
            return None
        try:
            dt = datetime.utcfromtimestamp(ts + tz_offset)
            return dt.strftime(fmt)
        except Exception:
            return None

    def _infer_time_scope(self, query: str, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if metadata:
            scope = (metadata.get("entities") or {}).get("time_scope")
            if isinstance(scope, str):
                return scope.lower()
        lowered = query.lower()
        if any(word in lowered for word in ["tomorrow", "明天", "翌日"]):
            return "tomorrow"
        if any(word in lowered for word in ["afternoon", "下午"]):
            return "afternoon"
        if any(word in lowered for word in ["evening", "tonight", "今晚", "晚上"]):
            return "evening"
        if any(word in lowered for word in ["sunset", "日落"]):
            return "sunset"
        if any(word in lowered for word in ["sunrise", "日出"]):
            return "sunrise"
        return None

    def _select_hourly_slot(self, hours: List[Dict[str, Any]], time_scope: Optional[str]) -> Optional[Dict[str, Any]]:
        if not hours:
            return None
        if not time_scope:
            return hours[0]

        def hour_of(entry: Dict[str, Any]) -> Optional[int]:
            try:
                return datetime.utcfromtimestamp(entry["time"]).hour
            except Exception:
                return None

        ranges = {
            "afternoon": range(12, 18),
            "evening": range(18, 24),
            "sunrise": range(5, 8),
            "sunset": range(17, 20),
        }

        target_range = ranges.get(time_scope)
        if target_range:
            for entry in hours:
                hour = hour_of(entry)
                if hour in target_range:
                    return entry
        if time_scope == "tomorrow" and len(hours) > 8:
            return hours[8]
        return hours[0]

    def _describe_period(self, time_scope: Optional[str]) -> str:
        mapping = {
            "afternoon": "下午",
            "evening": "傍晚",
            "sunrise": "日出时段",
            "sunset": "日落时段",
            "tomorrow": "明日",
        }
        return mapping.get(time_scope or "", "稍后")

    def _deduce_advisories(
        self,
        current: Dict[str, Any],
        hours: List[Dict[str, Any]],
        air_quality: Optional[Dict[str, Any]],
    ) -> List[str]:
        advisories: List[str] = []
        wind = current.get("wind_speed")
        if isinstance(wind, (int, float)) and wind >= 15:
            advisories.append("有较强阵风，户外活动请注意防风")
        if any("雨" in (h.get("description") or "") for h in hours[:4]):
            advisories.append("短时可能有降水，出门携带雨具")
        if air_quality and air_quality.get("aqi") and air_quality["aqi"] >= 4:
            advisories.append("空气质量偏差，敏感人群减少户外活动")
        return advisories

    def _map_aqi(self, value: Optional[int]) -> str:
        mapping = {1: "优", 2: "良", 3: "中等", 4: "较差", 5: "严重"}
        return mapping.get(value, "未知")





# 交通 API
class TrafficAPIHandler(BaseAPIHandler):
    def __init__(self):
        self.client = googlemaps.Client(key=os.getenv("TRAFFIC_API_KEY"))

    def parse_locations(self, query: str):
        patterns = [
            r"从(?P<origin>.+?)到(?P<destination>.+)",
            r"由(?P<origin>.+?)到(?P<destination>.+)",
            r"由(?P<origin>.+?)前往(?P<destination>.+)",
            r"(?P<origin>.+?)到(?P<destination>.+)",
            r"(?P<origin>.+?)去(?P<destination>.+)",
            r"route from (?P<origin>.+?) to (?P<destination>.+)",
            r"from (?P<origin>.+?) to (?P<destination>.+)",
        ]
        lowered = query.lower()
        for pattern in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if match:
                origin = match.group("origin").strip(" 。?？.")
                destination = match.group("destination").strip(" 。?？.")
                return origin, destination

        # handle "route from A to B" after removing leading text
        if "route" in lowered and "to" in lowered:
            try:
                idx = lowered.index("route")
                route_part = query[idx:]
                match = re.search(r"route .*? from (?P<origin>.+?) to (?P<destination>.+)", route_part, flags=re.IGNORECASE)
                if match:
                    return match.group("origin").strip(), match.group("destination").strip()
            except ValueError:
                pass

        return None, None

    def handle(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        origin, destination = self.parse_locations(query)

        if not origin or not destination:
            return {"error": "未能识别起点和终点，请使用'从A到B'或'route from A to B'的格式。"}

        try:
            directions = self.client.directions(
                origin=origin,
                destination=destination,
                mode="driving",
                departure_time="now"
            )
        except Exception as exc:
            return {"error": f"交通信息请求失败：{exc}"}

        if not directions:
            return {"error": "无法获取交通信息"}

        leg = directions[0]["legs"][0]
        duration = leg["duration"]["text"]
        distance = leg["distance"]["text"]
        steps = [
            {
                "instruction": self._strip_html(step.get("html_instructions", "")),
                "distance": step.get("distance", {}).get("text"),
                "duration": step.get("duration", {}).get("text"),
            }
            for step in leg.get("steps", [])
        ]

        summary = f"从 {leg['start_address']} 到 {leg['end_address']} 预计行驶 {duration}（{distance}）"
        return {
            "summary": summary,
            "mode": "driving",
            "distance": distance,
            "duration": duration,
            "start_address": leg.get("start_address"),
            "end_address": leg.get("end_address"),
            "steps": steps,
            "raw": directions,
        }

    @staticmethod
    def _strip_html(text: str) -> str:
        return re.sub(r"<.*?>", "", text)





# 金融 API
class FinanceAPIHandler(BaseAPIHandler):
    HSI_KEYWORDS = ["恒生指数", "hsi", "hang seng"]
    GOLD_KEYWORDS = ["gold", "gold price", "金价", "金價", "黃金", "黄金"]
    GOLD_TICKER = "XAUHKD=X"
    COMPARISON_TICKERS = ["NVDA", "AMD"]
    CLP_KEYWORDS = ["clp", "中电", "電價", "電費", "tariff"]

    def handle(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        normalized = query.lower()
        entities = (metadata or {}).get("entities") or {}
        entity_ticker = entities.get("ticker")
        entity_ticker_lower = entity_ticker.lower() if isinstance(entity_ticker, str) else ""

        if any(word in normalized for word in self.CLP_KEYWORDS):
            return {
                "error": "CLP 電費屬於公共服務資訊，請改用本地知識庫或公共服務 API 查詢。"
            }

        try:
            if self._is_hsi_request(normalized):
                return self._get_index_change("^HSI", "恒生指数")
            if self._is_gold_request(normalized, entity_ticker_lower):
                return self._get_gold_price()
            if self._is_comparison_request(normalized):
                return self._compare_stocks(self.COMPARISON_TICKERS)
            if self._is_fx_request(normalized):
                return self._get_fx_rate(normalized)
            return self._get_single_ticker(query, metadata)
        except Exception as exc:
            return {"error": f"金融数据获取失败：{exc}"}

    def _is_hsi_request(self, normalized: str) -> bool:
        return any(word in normalized for word in self.HSI_KEYWORDS)

    def _is_gold_request(self, normalized: str, entity_ticker: str) -> bool:
        if entity_ticker:
            if any(word in entity_ticker for word in self.GOLD_KEYWORDS):
                return True
        return any(word in normalized for word in self.GOLD_KEYWORDS)

    def _is_comparison_request(self, normalized: str) -> bool:
        return "nvda" in normalized and "amd" in normalized

    def _is_fx_request(self, normalized: str) -> bool:
        return ("hkd" in normalized and ("jpy" in normalized or "yen" in normalized)) or ("港币" in normalized and "日元" in normalized) or ("港幣" in normalized and "日元" in normalized)

    def _get_index_change(self, symbol: str, name: str) -> Dict[str, Any]:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        if hist.empty:
            raise ValueError("指数数据为空")
        close = hist["Close"]
        current = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) > 1 else current
        pct_change = ((current / prev) - 1) * 100 if prev else None
        summary = f"{name}收于 {current:.2f} 点，日变动 {pct_change:.2f}%"
        return {
            "summary": summary,
            "instrument": {"symbol": symbol, "name": name},
            "quote": {
                "last": current,
                "previous_close": prev,
                "pct_change": pct_change,
                "currency": "HKD",
                "timestamp": hist.index[-1].isoformat() if not hist.index.empty else None,
            },
            "raw": {"history": hist.to_dict()},
        }

    def _get_gold_price(self) -> Dict[str, Any]:
        ticker = yf.Ticker(self.GOLD_TICKER)
        hist = ticker.history(period="1d")
        if hist.empty:
            # fallback to XAU/USD + HKD/USD
            price = self._fallback_gold_price_hkd()
            summary = f"当前黄金价约 {price:.2f} HKD/盎司"
            return {
                "summary": summary,
                "instrument": {"symbol": self.GOLD_TICKER, "name": "Gold (XAU/HKD)"},
                "quote": {"last": price, "currency": "HKD", "timestamp": None},
                "raw": {"fallback": True},
            }
        price = float(hist["Close"].iloc[-1])
        summary = f"当前黄金价约 {price:.2f} HKD/盎司"
        return {
            "summary": summary,
            "instrument": {"symbol": self.GOLD_TICKER, "name": "Gold (XAU/HKD)"},
            "quote": {
                "last": price,
                "currency": "HKD",
                "timestamp": hist.index[-1].isoformat() if not hist.index.empty else None,
            },
            "raw": {"history": hist.to_dict()},
        }

    def _fallback_gold_price_hkd(self) -> float:
        gold_usd = yf.Ticker("XAUUSD=X").history(period="1d")
        if gold_usd.empty:
            gold_usd = yf.Ticker("GC=F").history(period="1d")
            if gold_usd.empty:
                raise ValueError("无法从美元获取黄金价格")
        usd_hkd = yf.Ticker("USDHKD=X").history(period="1d")
        if usd_hkd.empty:
            raise ValueError("无法转换为港币价格")
        price_usd = float(gold_usd["Close"].iloc[-1])
        rate = float(usd_hkd["Close"].iloc[-1])
        return price_usd * rate

    def _compare_stocks(self, symbols: List[str]) -> Dict[str, Any]:
        comparison = []
        best_symbol = None
        best_change = float("-inf")

        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if hist.empty or len(hist["Close"]) < 2:
                continue
            start = float(hist["Close"].iloc[0])
            end = float(hist["Close"].iloc[-1])
            pct_change = ((end / start) - 1) * 100
            if pct_change > best_change:
                best_change = pct_change
                best_symbol = symbol
            comparison.append(
                {
                    "symbol": symbol,
                    "start": start,
                    "end": end,
                    "pct_change": pct_change,
                    "timestamp_start": hist.index[0].isoformat(),
                    "timestamp_end": hist.index[-1].isoformat(),
                }
            )

        if not comparison:
            raise ValueError("无法获取比较所需的股票数据")

        summary = f"{best_symbol} 5 日表现最佳，涨跌幅 {best_change:.2f}%。"
        return {
            "summary": summary,
            "comparison": comparison,
            "raw": {"symbols": symbols},
        }

    def _get_fx_rate(self, normalized_query: str) -> Dict[str, Any]:
        amount = self._extract_amount(normalized_query)
        pair = "JPYHKD=X"
        ticker = yf.Ticker(pair)
        hist = ticker.history(period="1d")
        if hist.empty:
            raise ValueError("无法获取汇率")
        rate = float(hist["Close"].iloc[-1])
        converted = rate * amount if amount is not None else None
        summary = f"当前 JPY/HKD 汇率 {rate:.4f}"
        if converted is not None:
            summary += f"，{amount:.0f} 日元约合 {converted:.2f} 港币"
        return {
            "summary": summary,
            "fx": {
                "pair": "JPY/HKD",
                "rate": rate,
                "inverse_rate": (1 / rate) if rate else None,
                "amount_jpy": amount,
                "amount_hkd": converted,
                "timestamp": hist.index[-1].isoformat() if not hist.index.empty else None,
            },
            "raw": {"history": hist.to_dict()},
        }

    def _extract_amount(self, query: str) -> Optional[float]:
        match = re.search(r"([\d,]+)\s*(日元|yen|jpy)", query, re.IGNORECASE)
        if not match:
            return None
        value = match.group(1).replace(",", "")
        try:
            return float(value)
        except ValueError:
            return None

    def _get_single_ticker(self, query: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        ticker_symbol = None
        matched_name = None

        for name, symbol in TICKER_MAP.items():
            if name in query:
                ticker_symbol = symbol
                matched_name = name
                break

        if not ticker_symbol:
            search = yf.Search(query, max_results=3, enable_fuzzy_query=True)
            if not search.quotes:
                raise ValueError(f"未找到与 '{query}' 相关的股票代码")
            ticker_symbol = search.quotes[0]["symbol"]
            matched_name = search.quotes[0].get("shortname", ticker_symbol)

        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1d")
        if hist.empty:
            raise ValueError(f"未能获取 {matched_name} ({ticker_symbol}) 的最新数据")

        price = float(hist["Close"].iloc[-1])
        summary = f"{matched_name} ({ticker_symbol}) 最新收盘价 {price:.2f}"
        return {
            "summary": summary,
            "instrument": {"symbol": ticker_symbol, "name": matched_name},
            "quote": {
                "last": price,
                "currency": (ticker.fast_info.get("currency") if hasattr(ticker, "fast_info") else None),
                "timestamp": hist.index[-1].isoformat() if not hist.index.empty else None,
            },
            "raw": {"history": hist.to_dict()},
        }








# Google Search API
class GoogleSearchAPIHandler(BaseAPIHandler):
    ENDPOINT = "https://customsearch.googleapis.com/customsearch/v1"
    NEWS_KEYWORDS = [
        "新闻",
        "消息",
        "報道",
        "报道",
        "快訊",
        "快讯",
        "headlines",
        "headline",
        "breaking",
        "update",
        "updates",
        "latest news",
        "最新消息",
        "最新动态",
        "最新新聞",
        "news",
    ]
    REFERENCE_KEYWORDS = [
        "如何",
        "怎么",
        "怎麼",
        "怎么办",
        "指南",
        "指引",
        "政策",
        "規定",
        "规定",
        "assessment",
        "assess",
        "评估",
        "紫外",
        "紫外線",
        "紫外线",
        "uv",
        "空气质量",
        "空氣質量",
        "aqi",
        "风球",
        "風球",
        "typhoon",
        "signal",
        "热带气旋",
        "熱帶氣旋",
        "tropical cyclone",
    ]
    AUGMENT_RULES = [
        {
            "keywords": ["紫外線", "紫外线", "uv"],
            "extras": ['"UV index"', "紫外线 指数"],
        },
        {
            "keywords": ["风球", "風球", "typhoon signal", "热带气旋", "熱帶氣旋", "台风", "颱風"],
            "extras": ["香港 天文台", "HKO", '"typhoon signal"'],
        },
        {
            "keywords": ["空气质量", "空氣質量", "aqi"],
            "extras": ['"air quality"', "AQI"],
        },
    ]

    def __init__(self):
        self.api_key = os.getenv("GOOGLESEARCH_API_KEY")
        self.engine_id = os.getenv("GOOGLESEARCH_ENGINE_ID")
        self.session = requests.Session()

    def _format_items(self, items):
        lines = []
        for idx, item in enumerate(items[:5], 1):
            title = item.get("title") or "无标题"
            link = item.get("link") or ""
            snippet = (item.get("snippet") or "").replace("\n", " ")
            snippet = re.sub(r"\s+", " ", snippet).strip()
            lines.append(f"[{idx}] {title}\n链接: {link}\n摘要: {snippet}")
        return "\n\n".join(lines)

    def handle(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key or not self.engine_id:
            error_msg = "Google Search API 未配置，请设置 GOOGLESEARCH_API_KEY 及 GOOGLESEARCH_ENGINE_ID。"
            return {"summary": error_msg, "error": error_msg, "items": [], "meta": {"strategy": "config"}}

        search_query, intent_name, time_scope, site = self._extract_entities(query, metadata)
        strategy = self._determine_strategy(search_query, metadata, site)
        final_query = self._augment_query(search_query, strategy)
        params = self._build_params(final_query, time_scope, strategy, site)

        try:
            response = self.session.get(self.ENDPOINT, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            error_msg = f"Google Search API 请求失败：{exc}"
            return {
                "summary": error_msg,
                "error": error_msg,
                "items": [],
                "meta": {"strategy": strategy, "query": final_query},
            }

        items = data.get("items") or []
        if not items:
            error_msg = "Google Search 未找到相关结果。"
            return {
                "summary": error_msg,
                "error": error_msg,
                "items": [],
                "meta": {"strategy": strategy, "query": final_query},
                "raw": {"params": self._public_params(params), "response": data},
            }

        formatted_text = self._format_items(items)
        normalized_items = self._normalize_items(items)
        summary = f"Google搜索策略：{strategy}；查询：{final_query}\n{formatted_text}"

        return {
            "summary": summary,
            "items": normalized_items,
            "meta": {
                "strategy": strategy,
                "query": final_query,
                "site": site,
                "time_scope": time_scope,
            },
            "raw": {"params": self._public_params(params), "response": data},
        }

    def _normalize_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for idx, item in enumerate(items[:5], 1):
            normalized.append(
                {
                    "rank": idx,
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "displayLink": item.get("displayLink"),
                    "snippet": (item.get("snippet") or "").strip(),
                }
            )
        return normalized

    def _extract_entities(
        self, query: str, metadata: Optional[Dict[str, Any]]
    ) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        intent_name = None
        time_scope = None
        site = None
        search_query = query

        if metadata:
            intent_name = metadata.get("intent")
            entities = metadata.get("entities") or {}

            candidate_query = entities.get("search_query")
            if isinstance(candidate_query, str) and candidate_query.strip():
                search_query = candidate_query.strip()

            candidate_site = entities.get("site")
            if isinstance(candidate_site, str) and candidate_site.strip():
                site = candidate_site.strip()

            candidate_time_scope = entities.get("time_scope")
            if intent_name == "google_search_api" and isinstance(candidate_time_scope, str):
                time_scope = candidate_time_scope.strip()

        return search_query, intent_name, time_scope, site

    def _determine_strategy(
        self,
        query: str,
        metadata: Optional[Dict[str, Any]],
        site: Optional[str],
    ) -> str:
        if site:
            return "site"

        intent_name = metadata.get("intent") if metadata else None
        reason = (metadata or {}).get("reason") or ""
        if self._contains_keyword(query, self.NEWS_KEYWORDS) or self._contains_keyword(reason, self.NEWS_KEYWORDS):
            return "news"
        if intent_name == "google_search_api":
            entities = (metadata or {}).get("entities") or {}
            search_query = entities.get("search_query") or ""
            if self._contains_keyword(search_query, self.NEWS_KEYWORDS):
                return "news"

        if self._contains_keyword(query, self.REFERENCE_KEYWORDS):
            return "reference"

        return "general"

    def _augment_query(self, query: str, strategy: str) -> str:
        additions: List[str] = []
        lowered = query.lower()
        for rule in self.AUGMENT_RULES:
            if any(keyword.lower() in lowered for keyword in rule["keywords"]):
                additions.extend(rule["extras"])

        if strategy == "news" and "news" not in lowered and "新闻" not in query:
            additions.append("news")

        if not additions:
            return query

        unique_additions = list(dict.fromkeys(additions))
        return f"{query} " + " ".join(unique_additions)

    def _contains_keyword(self, text: str, keywords: List[str]) -> bool:
        if not text:
            return False
        lowered = text.lower()
        for keyword in keywords:
            if keyword in text or keyword.lower() in lowered:
                return True
        return False

    def _build_params(
        self,
        query: str,
        time_scope: Optional[str],
        strategy: str,
        site: Optional[str],
    ) -> dict:
        params = {
            "key": self.api_key,
            "cx": self.engine_id,
            "q": query,
            "num": 5,
            "hl": "zh-CN",
            "safe": "active",
        }

        if site:
            params["siteSearch"] = site

        normalized_scope = (time_scope or "").lower()
        if normalized_scope:
            date_params = self._params_from_time_scope(normalized_scope)
            if date_params:
                params.update(date_params)

        if strategy == "news":
            params.setdefault("sort", "date")
            params.setdefault("dateRestrict", "d7")
        elif strategy == "reference":
            params["num"] = 7

        return params

    @staticmethod
    def _params_from_time_scope(time_scope: str) -> Optional[Dict[str, str]]:
        mapping = {
            "today": "d1",
            "this_week": "d7",
            "past_week": "d7",
            "week": "d7",
            "recent": "d7",
            "this_month": "m1",
            "past_month": "m1",
            "month": "m1",
            "30d": "m1",
            "this_year": "y1",
            "past_year": "y1",
            "year": "y1",
        }

        date_restrict = mapping.get(time_scope)
        if not date_restrict and time_scope.startswith("custom:"):
            return None

        if date_restrict:
            return {"dateRestrict": date_restrict, "sort": "date"}

        return None

    @staticmethod
    def _public_params(params: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in params.items() if k not in {"key"}}


# 股票别名映射表（可扩展）
TICKER_MAP = {
    "中国石化": "600028.SS",
    "中石化": "600028.SS",
    "Sinopec": "600028.SS",
    "中国石油": "601857.SS",
    "中石油": "601857.SS",
    "腾讯": "0700.HK",
    "阿里巴巴": "BABA",
    "比亚迪": "002594.SZ",
    "茅台": "600519.SS"
}

# Router 维护映射表
HANDLERS = {
    "weather_api": WeatherAPIHandler(),
    "traffic_api": TrafficAPIHandler(),
    "finance_api": FinanceAPIHandler(),
    "google_search_api": GoogleSearchAPIHandler(),

}

