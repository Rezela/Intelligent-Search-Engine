import requests
import googlemaps
import re
import yfinance as yf


# 定义统一接口
class BaseAPIHandler:
    def handle(self, query: str) -> str:
        raise NotImplementedError


# 天气 API
class WeatherAPIHandler(BaseAPIHandler):
    def __init__(self):
        self.api_key = "8a235aa55a5ca1fab61e5e37d5fdf605"

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
        self.client = googlemaps.Client(key="AIzaSyDfYrKGJ5ina9R0A4xYjNeJY-8iqkrIhF4")

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

            # 取第一个匹配结果
            symbol = search.quotes[0]["symbol"]
            name = search.quotes[0].get("shortname", symbol)

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if hist.empty:
                return f"未能获取 {name} ({symbol}) 的最新数据"

            price = hist["Close"].iloc[-1]
            return f"{name} ({symbol}) 最新收盘价：{price:.2f}"
        except Exception as e:
            return f"金融数据获取失败：{str(e)}"

# Router 维护映射表
HANDLERS = {
    "weather_api": WeatherAPIHandler(),
    "traffic_api": TrafficAPIHandler(),
    "finance_api": FinanceAPIHandler(),
}