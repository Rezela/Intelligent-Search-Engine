ROUTING_RULES = {
    "weather": {
        "keywords": [
            "天气", "气温", "温度", "湿度", "风", "下雨", "晴天", "阴天", "多云",
            "暴雨", "台风", "空气质量", "天气预报"
        ],
        "api": "weather_api"
    },
    "traffic": {
        "keywords": [
            "交通", "路况", "拥堵", "堵车", "车程", "地铁", "公交", "路线", "开车",
            "骑车", "步行", "到…要多久", "到…距离"
        ],
        "api": "traffic_api"
    },
    "finance": {
        "keywords": [
            "股票", "股价", "基金", "指数", "恒生指数", "货币", "汇率", "外汇",
            "美元", "人民币", "港币", "coin", "crypto", "比特币", "以太坊",
            "market", "finance", "收盘价", "行情"
        ],
        "api": "finance_api"
    }
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
