import requests
from yfinance import yf
import googlemaps


# 定义统一接口
class BaseAPIHandler:
    def handle(self, query: str) -> str:
        raise NotImplementedError


# 天气 API
class WeatherAPIHandler(BaseAPIHandler):
    def __init__(self, api_key: str):
        self.api_key = "8a235aa55a5ca1fab61e5e37d5fdf605"
        
    def handle(self, query: str):
        # 假设 query 包含城市
        city = query.replace("天气", "").strip()
        if not city:
            city = "Hong Kong"

        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.api_key}&lang=zh_cn&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200 and "main" in data:
            temp = data["main"]["temp"]
            description = data["weather"][0]["description"]
            return f"{city} 当前气温 {temp} ℃，天气情况为 {description}"
        else:
            return f"无法获取 {city} 的天气信息"

class TrafficAPIHandler(BaseAPIHandler):
    def handle(self, query: str):
        return 0


class FinanceAPIHandler(BaseAPIHandler):
    def handle(self, query: str):
        return 0

# Router 维护映射表
HANDLERS = {
    "weather_api": WeatherAPIHandler(),
    "traffic_api": TrafficAPIHandler(),
    "finance_api": FinanceAPIHandler(),
}