import requests
import googlemaps
import re
import yfinance as yf
from datetime import datetime, timedelta

import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 定义统一接口
class BaseAPIHandler:
    def handle(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError





# 天气 API
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

    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")
        self.session = requests.Session()
        self.lang = os.getenv("WEATHER_LANG", "zh_cn")
        self.units = os.getenv("WEATHER_UNITS", "metric")
        self._geo_cache: Dict[str, Dict[str, Any]] = {}

    def handle(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key:
            return {"error": "未配置 WEATHER_API_KEY"}

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

        summary = self._build_summary(
            location["name"],
            current_detail,
            hourly_detail,
            daily_detail,
            air_detail,
            advisories,
            self._infer_time_scope(query, metadata),
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
            "raw": {
                "current": current_raw,
                "forecast_hourly": hourly_raw,
                "forecast_daily": daily_raw,
                "air_quality": air_raw,
                "air_quality_forecast": air_forecast_raw,
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
            return {"error": "未能识别起点和终点，请使用“从A到B”或“route from A to B”的格式。"}

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
    RECENT_KEYWORDS = ["最新", "今天", "今日", "近期", "最近", "news", "latest", "today", "recent"]

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

    def handle(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        if not self.api_key or not self.engine_id:
            return "Google Search API 未配置，请设置 GOOGLESEARCH_API_KEY 及 GOOGLESEARCH_ENGINE_ID。"

        time_scope = self._extract_time_scope(metadata)
        params = self._build_params(query, time_scope)

        try:
            response = self.session.get(self.ENDPOINT, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            return f"Google Search API 请求失败：{exc}"

        items = data.get("items") or []
        if not items:
            return "Google Search 未找到相关结果。"

        return self._format_items(items)

    def _extract_time_scope(self, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if not metadata:
            return None
        entities = metadata.get("entities") or {}
        time_scope = entities.get("time_scope")
        if isinstance(time_scope, str):
            return time_scope.strip()
        return None

    def _build_params(self, query: str, time_scope: Optional[str]) -> dict:
        params = {
            "key": self.api_key,
            "cx": self.engine_id,
            "q": query,
            "num": 5,
            "hl": "zh-CN",
            "safe": "active",
        }

        normalized = (time_scope or "").lower()
        if normalized:
            date_params = self._params_from_time_scope(normalized)
            if date_params:
                params.update(date_params)
        else:
            lowered = query.lower()
            if any(keyword in query for keyword in self.RECENT_KEYWORDS):
                params["sort"] = "date"
                params["dateRestrict"] = "d7"
            elif "本周" in query or "近一周" in query:
                params["dateRestrict"] = "d7"
                params["sort"] = "date"
            elif "本月" in query or "近一月" in query:
                params["dateRestrict"] = "m1"
                params["sort"] = "date"
            elif "今年" in query or "year" in lowered:
                params["dateRestrict"] = "y1"
                params["sort"] = "date"

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

