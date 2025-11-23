import logging
from typing import Optional


ROUTING_RULES = {
    "weather": {
        "keywords": [
            "天气",
            "气温",
            "温度",
            "湿度",
            "风",
            "下雨",
            "晴天",
            "阴天",
            "多云",
            "暴雨",
            "台风",
            "空气质量",
            "天气预报",
            "信号",
            "信號",
            "风球",
            "typhoon",
            "signal",
            "公共圖書館",
            "公共图书馆",
            "public library",
        ],
        "api": "weather_api",
    },
    "traffic": {
        "keywords": ["交通", "路况", "拥堵", "堵车", "地铁", "公交", "路线", "开车", "骑车", "步行", "多久", "要多久", "距离", "多远"],
        "api": "traffic_api",
    },
    "finance": {
        "keywords": ["股票", "股价", "基金", "指数", "恒生指数", "货币", "汇率", "外汇", "美元", "人民币", "港币", "收盘价", "行情", "coin", "crypto", "比特币", "以太坊", "market", "finance"],
        "api": "finance_api",
    },
    "google_search": {
        "keywords": ["google", "搜索", "网上资料", "最新消息", "新闻", "报道", "资讯", "网上查一下"],
        "api": "google_search_api",
    },
    "image_analysis": {
        "keywords": [
            "图片",
            "相片",
            "照片",
            "截图",
            "截圖",
            "图像",
            "圖像",
            "screenshot",
            "image",
            "photo",
        ],
        "api": "image_analysis_api",
    },
}


class SourceRouter:
    """
    优先使用 LLM 做意图识别，失败时回退到关键词规则。
    """

    def __init__(self, intent_classifier=None, rules=None):
        self.rules = rules or ROUTING_RULES
        self.intent_classifier = intent_classifier
        self._weather_forced_keywords = [
            "公共圖書館",
            "公共图书馆",
            "public library",
            "八號風球",
            "typhoon signal",
            "熱帶氣旋信號",
        ]

    def route(self, query: str, image_data: Optional[str] = None):
        if image_data:
            return "image_analysis_api"
        lowered = query.lower()
        if any(keyword.lower() in lowered for keyword in self._weather_forced_keywords):
            return "weather_api"

        if self.intent_classifier:
            try:
                intent = self.intent_classifier.select_intent(query)
                if intent:
                    return intent
            except Exception as exc:
                # LLM 分类出现异常时，继续使用关键词回退
                logging.warning("SourceRouter: LLM intent classifier 异常：%s", exc)
                pass

        for cfg in self.rules.values():
            if any(keyword.lower() in lowered for keyword in cfg["keywords"]):
                return cfg["api"]

        return "rag"
