"""
文件路径: api/routes/pages.py
功能: 页面路由 - 渲染HTML页面

提供Web界面的页面路由，使用Jinja2模板渲染。
"""

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path

# 创建路由器
router = APIRouter()

# 获取项目根目录和模板目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))


# ==================== 仪表盘 ====================

@router.get("/", name="dashboard")
async def dashboard(request: Request):
    """仪表盘首页"""
    return templates.TemplateResponse("pages/dashboard.html", {
        "request": request,
        "title": "仪表盘"
    })


# ==================== 记忆管理 ====================

@router.get("/memory", name="memory_list")
async def memory_list(request: Request):
    """记忆列表页"""
    return templates.TemplateResponse("pages/memory/list.html", {
        "request": request,
        "title": "记忆管理"
    })


@router.get("/memory/search", name="memory_search")
async def memory_search(request: Request):
    """记忆搜索页"""
    return templates.TemplateResponse("pages/memory/search.html", {
        "request": request,
        "title": "记忆搜索"
    })


# ==================== RAG知识库 ====================

@router.get("/rag", name="rag_documents")
async def rag_documents(request: Request):
    """文档管理页"""
    return templates.TemplateResponse("pages/rag/documents.html", {
        "request": request,
        "title": "文档管理"
    })


@router.get("/rag/search", name="rag_search")
async def rag_search(request: Request):
    """知识检索页"""
    return templates.TemplateResponse("pages/rag/search.html", {
        "request": request,
        "title": "知识检索"
    })


@router.get("/rag/chat", name="rag_chat")
async def rag_chat(request: Request):
    """问答对话页"""
    return templates.TemplateResponse("pages/rag/chat.html", {
        "request": request,
        "title": "知识问答"
    })


# ==================== 知识图谱 ====================

@router.get("/graph", name="graph_explorer")
async def graph_explorer(request: Request):
    """图谱浏览器页"""
    return templates.TemplateResponse("pages/graph/explorer.html", {
        "request": request,
        "title": "知识图谱"
    })


# ==================== 系统设置 ====================

@router.get("/settings", name="settings")
async def settings(request: Request):
    """系统设置页"""
    return templates.TemplateResponse("pages/settings.html", {
        "request": request,
        "title": "系统设置"
    })

