
ROUTING_RULES ={
    "weather": {
        "keywords": ["天气", "气温", "下雨", "温度", "湿度", "风"],
        "api": "weather_api"
    },
    "traffic": {
        "keywords": ["交通", "路况", "拥堵", "地铁", "公交"],
        "api": "traffic_api"
    },
    "finance": {
        "keywords": ["股票", "基金", "货币", "汇率", "coin", "market"],
        "api": "finance_api"
    },
}

# 智能原选择： 规则路由（关键词匹配）
class SourceRouter:
    def __init__(self):
        self.rules = ROUTING_RULES

    def route(self, query: str):
        for source, cfg in self.rules.items():
            # 匹配关键词
            if any(keyword in query for keyword in cfg["keywords"]):
                return cfg["api"]
        # 若无匹配词， 默认使用 RAG
        return "rag"
