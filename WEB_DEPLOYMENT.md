# FullRAG Web 前端部署指南

## 📋 环境要求

### 必需依赖

1. **Python 3.8+**
2. **已安装的依赖包**（见 `requirements.txt`）
3. **FastAPI**（需要单独安装，见下方）

### 额外需要安装的包

```bash
pip install fastapi
```

或者更新 `requirements.txt`，添加：
```
fastapi>=0.104.0
```

> **注意**：`uvicorn` 已经在 `requirements.txt` 中，无需重复安装。

## 🚀 快速启动

### 方式一：直接运行（开发模式）

```bash
python web_app.py
```

服务器将在 `http://localhost:8000` 启动。

### 方式二：使用 uvicorn（推荐）

```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000 --reload
```

参数说明：
- `--host 0.0.0.0`: 允许外部访问（默认只允许 localhost）
- `--port 8000`: 端口号
- `--reload`: 开发模式，代码变更自动重载

### 方式三：生产环境部署

```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000 --workers 4
```

> **注意**：生产环境建议：
> - 移除 `--reload`
> - 使用反向代理（Nginx）
> - 配置 HTTPS
> - 限制 CORS 来源（修改 `web_app.py` 中的 `allow_origins`）

## 📁 文件结构

```
NLP/
├── web_app.py              # FastAPI 后端应用
├── static/
│   └── index.html         # 前端界面
├── RAG.py                 # RAG 核心逻辑
├── DeepSearch.py          # 深度搜索模块
├── SourceAPI.py           # API 处理器
├── DB.py                  # 数据库连接
├── chroma_db/            # 向量数据库（持久化）
└── .env                   # 环境变量配置
```

## 🔧 配置说明

### 环境变量（.env）

确保 `.env` 文件包含以下配置：

```env
# LLM API
HKGAI_API_KEY=your_api_key_here

# 天气 API
WEATHER_API_KEY=your_openweather_api_key
WEATHER_LANG=zh_cn
WEATHER_UNITS=metric

# Google Search API
GOOGLESEARCH_API_KEY=your_google_search_api_key
GOOGLESEARCH_ENGINE_ID=your_search_engine_id

# Google Maps API（交通查询）
GOOGLE_MAPS_API_KEY=your_google_maps_api_key

# 意图识别置信度阈值（可选）
LLM_INTENT_CONFIDENCE=0.55
```

### 数据库初始化

确保 `chroma_db/` 目录存在且已加载向量数据。如果首次运行，需要先执行数据导入：

```bash
# 参考原有的数据导入流程
python your_data_import_script.py
```

## 🌐 访问界面

启动后，在浏览器中访问：

- **前端界面**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs（Swagger UI）
- **健康检查**: http://localhost:8000/api/health

## 📡 API 接口说明

### POST `/api/query`

发送查询请求。

**请求体**:
```json
{
  "query": "北京今天的天气怎么样？",
  "language": "Chinese",
  "session_id": "optional_session_id",
  "use_deep_search": true
}
```

**响应**:
```json
{
  "answer": "北京今天...",
  "source": "weather_api",
  "session_id": "generated_session_id",
  "timing": {
    "api": 1.234
  },
  "context_preview": "...",
  "attempt_history": [...]
}
```

### GET `/api/session/{session_id}/history`

获取会话历史记录。

### DELETE `/api/session/{session_id}`

删除指定会话。

### GET `/api/health`

健康检查接口。

## 🎨 前端功能

- ✅ 实时对话界面
- ✅ 消息历史记录
- ✅ 来源标签显示（天气/交通/金融/搜索/知识库）
- ✅ 响应时间显示
- ✅ 深度搜索开关
- ✅ 自动滚动
- ✅ 加载动画
- ✅ 错误处理

## 🐛 常见问题

### 1. 端口被占用

```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### 2. CORS 错误

如果前端部署在不同域名，修改 `web_app.py` 中的 CORS 配置：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://your-frontend-domain.com"],  # 替换为实际域名
    ...
)
```

### 3. 静态文件未找到

确保 `static/index.html` 文件存在，或检查 `static_dir` 路径是否正确。

### 4. RAG 引擎初始化失败

- 检查 `chroma_db/` 目录是否存在
- 确认数据库已加载数据
- 查看日志文件中的错误信息

## 📝 开发建议

### 添加新功能

1. **新增 API 路由**：在 `web_app.py` 中添加新的 `@app.route` 装饰器
2. **修改前端**：编辑 `static/index.html` 中的 HTML/CSS/JS
3. **样式调整**：修改 `<style>` 标签中的 CSS

### 性能优化

- 使用 Redis 存储会话（替代内存字典）
- 添加请求缓存
- 使用 WebSocket 实现流式响应
- 前端添加请求防抖

## 🔒 安全建议

1. **生产环境**：
   - 限制 CORS 来源
   - 使用 HTTPS
   - 添加身份验证（JWT）
   - 限制 API 调用频率

2. **环境变量**：
   - 不要将 `.env` 文件提交到 Git
   - 使用密钥管理服务（如 AWS Secrets Manager）

## 📚 相关文档

- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Uvicorn 文档](https://www.uvicorn.org/)
- [ChromaDB 文档](https://docs.trychroma.com/)

