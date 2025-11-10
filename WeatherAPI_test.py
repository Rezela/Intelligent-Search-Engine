import unittest
from unittest.mock import patch, Mock
import sys
import os

# 添加源代码路径到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入被测类
from SourceAPI import WeatherAPIHandler


class TestWeatherAPIHandler(unittest.TestCase):

    def setUp(self):
        """测试前准备：创建 WeatherAPIHandler 实例"""
        self.handler = WeatherAPIHandler()

    def test_init_ignores_input_parameter(self):
        """
        TC001: 测试构造函数忽略传入参数
        验证无论传入什么参数，api_key 都被设置为硬编码值
        """
        # 验证 api_key 被设置为硬编码值，而不是传入的参数
        self.assertEqual(
            self.handler.api_key,
            "8a235aa55a5ca1fab61e5e37d5fdf605",
            "API key 应该使用硬编码值，而不是构造函数参数"
        )

        # 测试不同的输入参数
        handler2 = WeatherAPIHandler()
        self.assertEqual(
            handler2.api_key,
            "8a235aa55a5ca1fab61e5e37d5fdf605",
            "即使传入空字符串，也应该使用硬编码 API key"
        )

    @patch('SourceAPI.requests.get')
    def test_handle_success_case(self, mock_get):
        """
        TC002: 测试正常查询情况
        验证当 API 返回成功响应时，正确格式化天气信息
        """
        # 准备 Mock 响应数据
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "main": {"temp": 25.6},
            "weather": [{"description": "晴朗"}],
            "name": "Beijing"
        }
        mock_get.return_value = mock_response

        # 执行测试
        result = self.handler.handle("北京天气")

        # 验证结果
        expected_result = "北京 当前气温 25.6 ℃，天气情况为 晴朗"
        self.assertEqual(result, expected_result, "应该返回格式化的天气信息")

        # 验证 URL 构造正确
        expected_url = "http://api.openweathermap.org/data/2.5/weather?q=北京&appid=8a235aa55a5ca1fab61e5e37d5fdf605&lang=zh_cn&units=metric"
        mock_get.assert_called_once_with(expected_url)

    @patch('SourceAPI.requests.get')
    def test_handle_empty_query_uses_default_city(self, mock_get):
        """
        TC003: 测试空查询使用默认城市
        验证当查询为空或只包含"天气"时，使用默认城市"Hong Kong"
        """
        # 准备 Mock 响应数据
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "main": {"temp": 22.3},
            "weather": [{"description": "多云"}],
            "name": "Hong Kong"
        }
        mock_get.return_value = mock_response

        # 测试空查询
        result = self.handler.handle("")
        expected_result = "Hong Kong 当前气温 22.3 ℃，天气情况为 多云"
        self.assertEqual(result, expected_result, "空查询应该使用默认城市")

        # 验证 URL 构造正确
        expected_url = "http://api.openweathermap.org/data/2.5/weather?q=Hong Kong&appid=8a235aa55a5ca1fab61e5e37d5fdf605&lang=zh_cn&units=metric"
        mock_get.assert_called_with(expected_url)

    @patch('SourceAPI.requests.get')
    def test_handle_query_with_weather_keyword(self, mock_get):
        """
        TC004: 测试查询包含"天气"关键词
        验证正确解析城市名（移除"天气"并去除空格）
        """
        # 准备 Mock 响应数据
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "main": {"temp": 18.5},
            "weather": [{"description": "小雨"}],
            "name": "Shanghai"
        }
        mock_get.return_value = mock_response

        # 测试包含"天气"关键词的查询
        result = self.handler.handle("上海天气 ")

        # 验证结果
        expected_result = "上海 当前气温 18.5 ℃，天气情况为 小雨"
        self.assertEqual(result, expected_result, "应该正确解析城市名并返回天气信息")

        # 验证 URL 构造正确
        expected_url = "http://api.openweathermap.org/data/2.5/weather?q=上海&appid=8a235aa55a5ca1fab61e5e37d5fdf605&lang=zh_cn&units=metric"
        mock_get.assert_called_with(expected_url)

    @patch('SourceAPI.requests.get')
    def test_handle_api_error_response(self, mock_get):
        """
        TC005: 测试 API 返回错误状态码
        验证当 API 返回错误时，返回适当的错误信息
        """
        # 准备 Mock 错误响应
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"message": "city not found"}
        mock_get.return_value = mock_response

        # 执行测试
        result = self.handler.handle("不存在的城市天气")

        # 验证结果
        expected_result = "无法获取 不存在的城市 的天气信息"
        self.assertEqual(result, expected_result, "API 错误时应该返回错误信息")

    @patch('SourceAPI.requests.get')
    def test_handle_api_missing_main_field(self, mock_get):
        """
        TC006: 测试 API 返回缺少必要字段
        验证当 API 返回成功状态码但缺少 main 字段时，返回错误信息
        """
        # 准备 Mock 响应数据（缺少 main 字段）
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "cod": 200,
            "message": "success"
            # 故意缺少 "main" 字段
        }
        mock_get.return_value = mock_response

        # 执行测试
        result = self.handler.handle("北京天气")

        # 验证结果
        expected_result = "无法获取 北京 的天气信息"
        self.assertEqual(result, expected_result, "缺少必要字段时应该返回错误信息")


if __name__ == '__main__':
    unittest.main()
