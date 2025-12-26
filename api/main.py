"""
文件路径: api/main.py
功能: FastAPI 应用入口

主要功能:
- 创建 FastAPI 应用实例
- 配置 CORS 中间件
- 注册所有路由
- 配置全局异常处理
- 提供健康检查端点
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import sys
import os

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    启动时初始化资源，关闭时清理资源。
    """
    # 启动时
    logger.info("🚀 Agent Memory System API 启动中...")
    logger.info("📖 API文档: http://localhost:8000/docs")
    yield
    # 关闭时
    logger.info("👋 Agent Memory System API 关闭")


def create_app() -> FastAPI:
    """
    创建并配置 FastAPI 应用
    
    Returns:
        FastAPI: 配置完成的应用实例
    """
    settings = get_settings()
    
    # 创建应用
    app = FastAPI(
        title="Agent Memory System API",
        description="""
## 智能体记忆系统 - 可视化管理平台 API

提供以下功能模块:
- **记忆管理**: 增删改查、搜索、整合、遗忘
- **RAG知识库**: 文档上传、知识检索、问答
- **知识图谱**: 实体查询、路径查找、可视化
- **分析统计**: 仪表盘、趋势分析、系统监控
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # ==================== 配置 CORS ====================
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应限制为特定域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ==================== 挂载静态文件 ====================
    static_dir = BASE_DIR / "web" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info(f"📁 静态文件目录: {static_dir}")
    
    # ==================== 注册API路由 ====================
    from api.routes import memory, rag, graph, analytics
    
    app.include_router(
        memory.router, 
        prefix="/api/v1/memory", 
        tags=["记忆管理"]
    )
    app.include_router(
        rag.router, 
        prefix="/api/v1/rag", 
        tags=["RAG知识库"]
    )
    app.include_router(
        graph.router, 
        prefix="/api/v1/graph", 
        tags=["知识图谱"]
    )
    app.include_router(
        analytics.router, 
        prefix="/api/v1/analytics", 
        tags=["分析统计"]
    )
    
    # ==================== 注册页面路由 ====================
    from api.routes import pages
    app.include_router(pages.router, tags=["页面"])
    
    # ==================== 全局异常处理 ====================
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """全局异常处理器"""
        logger.error(f"未处理的异常: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "服务器内部错误",
                "detail": str(exc)
            }
        )
    
    # ==================== 系统端点 ====================
    @app.get("/health", tags=["系统"])
    async def health_check():
        """
        健康检查端点
        
        用于负载均衡器和监控系统检测服务状态。
        """
        return {
            "status": "ok",
            "service": "Agent Memory System",
            "version": "1.0.0"
        }
    
    # 注意：根路径 "/" 已由页面路由处理，用于显示仪表盘
    
    return app


# 创建应用实例
app = create_app()


# ==================== 启动入口 ====================
if __name__ == "__main__":
    import uvicorn
    
    # 从环境变量或默认值获取配置
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║           Agent Memory System API Server                 ║
╠══════════════════════════════════════════════════════════╣
║  启动地址: http://{host}:{port}                          
║  API文档:  http://localhost:{port}/docs                  
║  ReDoc:    http://localhost:{port}/redoc                 
╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload
    )

