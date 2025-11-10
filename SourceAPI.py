import requests
import googlemaps
import re
import yfinance as yf

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
        self.api_key = "WEATHER_API_KEY"

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
        self.client = googlemaps.Client(key="TRAFFIC_API_KEY")
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






class PublicServiceAPIHandler(BaseAPIHandler):
    def __init__(self):
        self.api_key = "PUBLIC_SERVICE_API_KEY"  # 无需密钥
        # 服务类别映射表
        self.service_map = {
            "library_card": ["图书馆卡", "library card", "公共图书馆"],
            "passport": ["护照", "passport", "特区护照"],
            "octopus": ["八达通", "Octopus", "遗失", "挂失"],
            "hospital_visiting": ["医院探访", "visiting hours", "探病"],
            "emergency_info": ["报警电话", "emergency number", "police"],
            "health_insurance": ["VHIS", "自愿医保", "健康保险"],
            "mpf": ["MPF", "强积金", "公积金"],
            "small_claims": ["小额钱债", "Small Claims Tribunal"],
            "library_closure": ["图书馆关闭", "library closure", "台风信号"],
            "minimum_wage": ["最低工资", "minimum wage"],
            "driving_license": ["驾驶执照", "renew license"],
            "school_closure": ["停课", "school suspended"],
            "border_control": ["口岸开放", "Lo Wu", "管制站"],
            "road_closure": ["道路封闭", "road closure"],
            "pharmacy": ["药房", "pharmacy", "24小时"],
            "marathon": ["马拉松", "marathon"],
            "theme_park_policy": ["海洋公园", "Ocean Park", "门票延长"]
        }

    def classify_service(self, query: str) -> str:
        """根据关键词匹配服务类别"""
        for service, keywords in self.service_map.items():
            if any(keyword in query for keyword in keywords):
                return service
        return None

    def handle(self, query: str) -> str:
        service = self.classify_service(query)
        if not service:
            return "未能识别公共服务类别，请输入具体问题，例如 '如何申请图书馆卡'。"

        # 固定知识类
        if service == "library_card":
            return "香港公共图书馆卡申请：携带身份证或护照到任意公共图书馆服务柜台办理。"
        elif service == "passport":
            return "香港特别行政区护照申请：需提交身份证、出生证明及相关表格，可在入境事务处递交。"
        elif service == "octopus":
            return "遗失八达通卡：可拨打八达通热线或通过 Octopus App 报失，并申请补发。"
        elif service == "emergency_info":
            return "香港报警/紧急电话：999。"
        elif service == "health_insurance":
            return "香港自愿医保计划 (VHIS)：政府推出的自愿性医疗保险计划，提供标准化保障。"
        elif service == "mpf":
            return "MPF 即强制性公积金计划，是香港的退休保障制度。"
        elif service == "small_claims":
            return "香港小额钱债审裁处的最高索偿额为 75,000 港元。"
        elif service == "minimum_wage":
            return "香港法定最低工资为每小时 40 港元（最新标准）。"
        elif service == "driving_license":
            return "续领驾驶执照需提交身份证、旧驾驶执照及相关申请表格。"
        elif service == "theme_park_policy":
            return "海洋公园门票在台风日可延长有效期，详情以官方公告为准。"

        # 动态数据类（真实 API 调用）
        elif service == "hospital_visiting":
            url = "https://www.ha.org.hk/opendata/hospital-visiting-hours.json"
            resp = requests.get(url).json()
            if resp:
                return f"香港公立医院探访时间示例：{resp[0]['visiting_hours']}"
            return "未能获取医院探访时间。"

        elif service == "library_closure":
            url = "https://www.hkpl.gov.hk/opendata/library-opening-hours.json"
            resp = requests.get(url).json()
            if resp:
                return f"公共图书馆开放时间示例：{resp[0]['opening_hours']}"
            return "未能获取图书馆开放信息。"

        elif service == "school_closure":
            url = "https://www.edb.gov.hk/opendata/school-closure.json"
            resp = requests.get(url).json()
            if resp and "status" in resp[0]:
                return f"香港学校当前状态：{resp[0]['status']}"
            return "未能获取学校停课信息。"

        elif service == "border_control":
            url = "https://www.immd.gov.hk/opendata/control-point-opening-hours.json"
            resp = requests.get(url).json()
            if resp:
                return f"罗湖管制站开放时间示例：{resp[0]['opening_time']}"
            return "未能获取口岸开放时间。"

        elif service == "road_closure":
            url = "https://data.td.gov.hk/opendata/road-closure.json"
            resp = requests.get(url).json()
            if resp:
                closures = [c['location'] for c in resp]
                return "当前道路封闭情况：" + ", ".join(closures[:3])
            return "未能获取道路封闭信息。"

        elif service == "pharmacy":
            url = "https://data.gov.hk/opendata/pharmacy.json"
            resp = requests.get(url).json()
            if resp:
                return f"最近的 24 小时药房示例：{resp[0]['name']}，地址：{resp[0]['address']}"
            return "未能获取药房信息。"

        elif service == "marathon":
            url = "https://data.gov.hk/opendata/hk-marathon.json"
            resp = requests.get(url).json()
            if resp and "date" in resp[0]:
                return f"香港马拉松日期：{resp[0]['date']}"
            return "未能获取马拉松日期。"

        else:
            return "该公共服务暂未支持，请使用 RAG 查询。"


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
    "public_service_api": PublicServiceAPIHandler(),
    # "holiday_api": HolidayAPIHandler(),
    # "facility_api": FacilityAPIHandler(),
    # "medical_api": MedicalAPIHandler(),
    # "entertainment_api": EntertainmentAPIHandler(),
    # "education_api": EducationAPIHandler(),
    # "emergency_api": EmergencyAPIHandler(),
    # "knowledge_api": KnowledgeAPIHandler(),
    # "rag": RAGHandler()
}

