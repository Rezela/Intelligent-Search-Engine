import requests
import googlemaps
import re
import yfinance as yf

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 定义统一接口
class BaseAPIHandler:
    def handle(self, query: str) -> str:
        raise NotImplementedError





# 天气 API
class WeatherAPIHandler(BaseAPIHandler):
    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")

    def extract_city(self, query: str) -> str:
        """
        简单地名提取：用正则匹配常见城市名
        可以扩展为更复杂的 NLP 模型或词典映射
        """
        # 常见城市列表，可扩展
        cities = ["北京", "上海", "广州", "深圳", "香港", "巴黎", "伦敦", "纽约"]
        for city in cities:
            if city in query:
                return city
        return None

    def geocode_city(self, city: str):
        """
        调用 OpenWeather Geocoding API，把城市名转成经纬度
        """
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={self.api_key}"
        resp = requests.get(url).json()
        if resp:
            lat, lon = resp[0]["lat"], resp[0]["lon"]
            return lat, lon
        return None, None

    def handle(self, query: str) -> str:
        # Step 1: 提取城市
        city = self.extract_city(query)
        if not city:
            return "未能识别城市，请输入类似 '北京天气' 的查询。"

        # Step 2: 获取经纬度
        lat, lon = self.geocode_city(city)
        if not lat or not lon:
            return f"无法解析 {city} 的地理位置"

        # Step 3: 调用天气 API
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.api_key}&lang=zh_cn&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200 and "main" in data:
            temp = data["main"]["temp"]
            description = data["weather"][0]["description"]
            return f"{city} 当前气温 {temp} ℃，天气情况为 {description}"
        else:
            return f"无法获取 {city} 的天气信息"





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

    def handle(self, query: str) -> str:
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
    def handle(self, query: str) -> str:
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
    RECENT_KEYWORDS = ["最新", "今天", "今日", "近期", "最近", "news", "latest", "today"]

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

    def handle(self, query: str) -> str:
        if not self.api_key or not self.engine_id:
            return "Google Search API 未配置，请设置 GOOGLESEARCH_API_KEY 及 GOOGLESEARCH_ENGINE_ID。"

        params = self._build_params(query)

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

    def _build_params(self, query: str) -> dict:
        params = {
            "key": self.api_key,
            "cx": self.engine_id,
            "q": query,
            "num": 5,
            "hl": "zh-CN",
            "safe": "active",
        }

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

