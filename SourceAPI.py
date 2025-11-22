import requests
import googlemaps
import re
import yfinance as yf

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
    FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
    AIR_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
    FALLBACK_CITIES = ["北京", "上海", "广州", "深圳", "香港", "巴黎", "伦敦", "纽约"]

    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")
        self.session = requests.Session()
        self.lang = os.getenv("WEATHER_LANG", "zh_cn")
        self.units = os.getenv("WEATHER_UNITS", "metric")

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

        forecast_raw = self._fetch_forecast(lat, lon)
        air_raw = self._fetch_air_quality(lat, lon)

        current_detail = self._extract_current(current_raw)
        forecast_detail = self._extract_forecast(forecast_raw)
        air_detail = self._extract_air_quality(air_raw)

        summary = self._build_summary(location["name"], current_detail, forecast_detail, air_detail)

        return {
            "summary": summary,
            "location": {
                "name": location["name"],
                "country": location.get("country"),
                "lat": lat,
                "lon": lon,
            },
            "current": current_detail,
            "forecast": forecast_detail,
            "air_quality": air_detail,
            "raw": {
                "current": current_raw,
                "forecast": forecast_raw,
                "air_quality": air_raw,
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
            if entity_city:
                return entity_city.strip()

        for city in self.FALLBACK_CITIES:
            if city in query:
                return city
        return None

    def _geocode_city(self, city: str) -> Optional[Dict[str, Any]]:
        params = {"q": city, "limit": 1}
        data = self._request(self.GEO_URL, params)
        if data:
            record = data[0]
            return {
                "name": record.get("name", city),
                "country": record.get("country"),
                "lat": record.get("lat"),
                "lon": record.get("lon"),
            }
        return None

    def _fetch_current_weather(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        params = {"lat": lat, "lon": lon, "units": self.units, "lang": self.lang}
        return self._request(self.CURRENT_URL, params)

    def _fetch_forecast(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        params = {"lat": lat, "lon": lon, "units": self.units, "lang": self.lang, "cnt": 8}
        return self._request(self.FORECAST_URL, params)

    def _fetch_air_quality(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        params = {"lat": lat, "lon": lon}
        return self._request(self.AIR_URL, params)

    def _extract_current(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        weather = (payload.get("weather") or [{}])[0]
        wind = payload.get("wind") or {}
        main = payload.get("main") or {}
        return {
            "description": weather.get("description"),
            "temperature": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "pressure": main.get("pressure"),
            "wind_speed": wind.get("speed"),
            "wind_direction": wind.get("deg"),
            "visibility": payload.get("visibility"),
            "sunrise": (payload.get("sys") or {}).get("sunrise"),
            "sunset": (payload.get("sys") or {}).get("sunset"),
            "timestamp": payload.get("dt"),
        }

    def _extract_forecast(self, payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not payload:
            return []
        entries = payload.get("list") or []
        highlights = []
        for item in entries[:4]:
            weather = (item.get("weather") or [{}])[0]
            main = item.get("main") or {}
            highlights.append(
                {
                    "time": item.get("dt"),
                    "temperature": main.get("temp"),
                    "description": weather.get("description"),
                    "humidity": main.get("humidity"),
                    "wind_speed": (item.get("wind") or {}).get("speed"),
                }
            )
        return highlights

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
            "components": components,
            "timestamp": records[0].get("dt"),
        }

    def _build_summary(
        self,
        city: str,
        current: Dict[str, Any],
        forecast: List[Dict[str, Any]],
        air_quality: Optional[Dict[str, Any]],
    ) -> str:
        desc = current.get("description") or "天气数据"
        temp = current.get("temperature")
        humidity = current.get("humidity")
        wind = current.get("wind_speed")
        pieces = [f"{city}当前{desc}"]
        if temp is not None:
            pieces.append(f"气温 {temp}°C")
        if humidity is not None:
            pieces.append(f"湿度 {humidity}%")
        if wind is not None:
            pieces.append(f"风速 {wind} m/s")
        if forecast:
            next_desc = forecast[0].get("description")
            next_temp = forecast[0].get("temperature")
            if next_desc or next_temp is not None:
                pieces.append(
                    f"未来几小时预计 {next_desc or '天气变化'}"
                    + (f"，温度约 {next_temp}°C" if next_temp is not None else "")
                )
        if air_quality and air_quality.get("aqi"):
            pieces.append(f"空气质量指数 AQI={air_quality['aqi']}")
        return "，".join(pieces)





# 交通 API
class TrafficAPIHandler(BaseAPIHandler):
    def __init__(self):
        self.client = googlemaps.Client(key=os.getenv("TRAFFIC_API_KEY"))
    def parse_locations(self, query: str):
        """
        使用正则表达式解析 query 中的起点和终点
        支持格式：
        - 从A到B
        - A到B
        - A去B
        - A到B要多久
        """
        patterns = [
            r"从(?P<origin>.+?)到(?P<destination>.+)",  # 从A到B
            r"(?P<origin>.+?)到(?P<destination>.+)",  # A到B
            r"(?P<origin>.+?)去(?P<destination>.+)"  # A去B
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                origin = match.group("origin").strip()
                destination = match.group("destination").strip()
                return origin, destination

        return None, None

    def handle(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        origin, destination = self.parse_locations(query)

        if not origin or not destination:
            return "未能识别起点和终点，请输入类似 '从香港科技大学到中环' 的格式。"

        directions = self.client.directions(
            origin=origin,
            destination=destination,
            mode="driving",
            departure_time="now"
        )

        if directions:
            leg = directions[0]["legs"][0]
            duration = leg["duration"]["text"]
            distance = leg["distance"]["text"]
            return f"从 {origin} 到 {destination} 预计车程 {duration}，距离 {distance}"
        else:
            return "无法获取交通信息"





# 金融 API
class FinanceAPIHandler(BaseAPIHandler):
    def handle(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        try:
            # Step 1: 尝试从别名映射表中匹配
            ticker_symbol = None
            for name, symbol in TICKER_MAP.items():
                if name in query:
                    ticker_symbol = symbol
                    matched_name = name
                    break

            # Step 2: 如果没有匹配到，再使用 yfinance.Search
            if not ticker_symbol:
                search = yf.Search(query, max_results=3, enable_fuzzy_query=True)
                """
                Search 类的源码里，返回的数据被解析到：
                self._quotes → 股票搜索结果（包含 symbol、shortname 等）
                self._news → 新闻结果
                self._lists → 列表结果
                self._research → 研究报告
                self._nav → 导航数据
                """
                if not search.quotes:
                    return f"未找到与 '{query}' 相关的股票代码"

                ticker_symbol = search.quotes[0]["symbol"]
                matched_name = search.quotes[0].get("shortname", ticker_symbol)

            # Step 3: 获取收盘价
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="1d")
            if hist.empty:
                return f"未能获取 {matched_name} ({ticker_symbol}) 的最新数据"

            price = hist["Close"].iloc[-1]
            return f"{matched_name} ({ticker_symbol}) 最新收盘价：{price:.2f}"

        except Exception as e:
            return f"金融数据获取失败：{str(e)}"








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

