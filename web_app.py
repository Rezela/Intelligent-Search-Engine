"""
FastAPI Web 应用入口
包装 FullRAG 为 RESTful API，支持会话管理和流式响应
"""
import os
import json
import time
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from DB import get_db
from RAG import FullRAG
from DeepSearch import DeepSearchManager
from logs import init_logger, new_query_id, log_user_query, log_assistant_answer
from document_loader import load_document_from_bytes, UnsupportedDocumentError
from build_index import ingest_text

# 全局状态：RAG 引擎和会话管理
rag_engine: Optional[FullRAG] = None
deep_search_manager: Optional[DeepSearchManager] = None
sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> {history, created_at}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时初始化 RAG，关闭时清理"""
    global rag_engine, deep_search_manager
    
    # 启动时初始化
    log_file = init_logger("web_app")
    logging.info("Web logger initialized at %s", log_file)
    logging.info("正在初始化 RAG 引擎...")
    try:
        db = get_db(persistent=True, path="./chroma_db", name="default")
        rag_engine = FullRAG(db)
        deep_search_manager = DeepSearchManager(rag_engine)
        logging.info("RAG 引擎初始化完成")
    except Exception as e:
        logging.error(f"RAG 引擎初始化失败: {e}")
        raise
    
    yield
    
    # 关闭时清理（可选）
    logging.info("正在关闭应用...")


app = FastAPI(
    title="FullRAG Web API",
    description="智能问答系统 - 支持天气、交通、金融、搜索和本地知识库",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务（前端 HTML/CSS/JS）
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ==================== 数据模型 ====================

class QueryRequest(BaseModel):
    query: str
    language: str = "Chinese"
    session_id: Optional[str] = None
    use_deep_search: bool = True
    image_data: Optional[str] = None  # 添加图片数据字段
    use_memory: bool = False


class QueryResponse(BaseModel):
    answer: str
    source: str
    session_id: str
    timing: Dict[str, float]
    context_preview: Optional[str] = None
    attempt_history: Optional[list] = None


class UploadDocResponse(BaseModel):
    session_id: str
    collection: str
    chunks: int
    persistent: bool


# ==================== API 路由 ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端 HTML 页面"""
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>FullRAG API</h1><p>前端页面未找到，请检查 static/index.html</p>"


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    处理用户查询
    
    - **query**: 用户问题
    - **language**: 回答语言（默认 Chinese）
    - **session_id**: 会话 ID（可选，用于多轮对话）
    - **use_deep_search**: 是否使用深度搜索（默认 True）
    """
    if not rag_engine or not deep_search_manager:
        raise HTTPException(status_code=503, detail="RAG 引擎未初始化")
    
    # 会话管理
    session_id = request.session_id or new_query_id()
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "created_at": time.time(),
            "temp_collections": [],
        }
    
    temp_collection_names = sessions[session_id].get("temp_collections", [])
    extra_collections = []
    for name in temp_collection_names:
        try:
            extra_collections.append(get_db(persistent=False, name=name))
        except Exception as exc:
            logging.warning("Unable to load temp collection %s: %s", name, exc)

    conversation_history = None
    if request.use_memory:
        history_items = sessions[session_id]["history"]
        if history_items:
            conversation_history = history_items[-5:]

    try:
        # 执行查询
        if request.use_deep_search:
            result = deep_search_manager.run(
                request.query,
                language=request.language,
                image_data=request.image_data,
                extra_collections=extra_collections,
                conversation_history=conversation_history,
            )
        else:
            result = rag_engine.query(
                request.query,
                language=request.language,
                image_data=request.image_data,
                extra_collections=extra_collections,
                conversation_history=conversation_history,
            )

        log_user_query(
            source="web_app",
            query=request.query,
            session_id=session_id,
            has_image=bool(request.image_data),
            extra={
                "use_deep_search": request.use_deep_search,
                "use_memory": request.use_memory,
                "source_used": result.get("source"),
            },
        )
        
        # 提取上下文预览
        context_preview = None
        context = result.get("context")
        if isinstance(context, dict):
            context_preview = context.get("summary") or json.dumps(
                {k: v for k, v in context.items() if k != "raw"},
                ensure_ascii=False,
            )[:200]
        elif isinstance(context, str):
            context_preview = context[:200]
        
        # 保存到会话历史
        sessions[session_id]["history"].append({
            "query": request.query,
            "answer": result.get("answer", ""),
            "source": result.get("source", "unknown"),
            "timestamp": time.time(),
        })
        
        log_assistant_answer(
            source="web_app",
            answer=result.get("answer", ""),
            session_id=session_id,
            timing=result.get("timing"),
            context_preview=context_preview,
            attempt_history=result.get("attempt_history"),
        )
        
        return QueryResponse(
            answer=result.get("answer", ""),
            source=result.get("source", "unknown"),
            session_id=session_id,
            timing=result.get("timing", {}),
            context_preview=context_preview,
            attempt_history=result.get("attempt_history"),
        )
    
    except Exception as e:
        logging.error(f"查询处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str):
    """获取指定会话的历史记录"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {
        "session_id": session_id,
        "history": sessions[session_id]["history"],
        "created_at": sessions[session_id]["created_at"],
        "temp_collections": sessions[session_id].get("temp_collections", []),
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "会话已删除"}
    raise HTTPException(status_code=404, detail="会话不存在")


@app.delete("/api/session/{session_id}/temp_collections")
async def clear_temp_collections(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    sessions[session_id]["temp_collections"] = []
    return {"message": "临时记忆已清空"}


@app.get("/api/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy" if rag_engine else "unhealthy",
        "sessions_count": len(sessions),
    }


@app.post("/api/upload_doc", response_model=UploadDocResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    language: str = Form("Chinese"),
    chunk_size: int = Form(300),
    persistent: bool = Form(False),
    collection: Optional[str] = Form(None),
):
    if chunk_size <= 0:
        raise HTTPException(status_code=400, detail="chunk_size must be positive")

    session_id = session_id or new_query_id()
    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "created_at": time.time(),
            "temp_collections": [],
        }

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空")

    try:
        text = load_document_from_bytes(file.filename or "uploaded", data)
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        logging.error("Document decode failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="无法解析上传文件")

    if not text.strip():
        raise HTTPException(status_code=400, detail="文档未提取到有效文本")

    try:
        collection_name, chunk_count = ingest_text(
            text=text,
            language=language,
            chunk_size=chunk_size,
            persistent=persistent,
            collection=collection,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        logging.error("Document ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="文档入库失败")

    if not persistent:
        temp_list = sessions[session_id].setdefault("temp_collections", [])
        if collection_name not in temp_list:
            temp_list.append(collection_name)

    return UploadDocResponse(
        session_id=session_id,
        collection=collection_name,
        chunks=chunk_count,
        persistent=persistent,
    )


if __name__ == "__main__":
    import uvicorn
    
    # 初始化日志
    init_logger()
    
    # 启动服务器
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式：代码变更自动重载
        log_level="info",
    )

