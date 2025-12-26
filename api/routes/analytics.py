"""
文件路径: api/routes/analytics.py
功能: 分析统计API路由

提供仪表盘数据、趋势分析和系统健康检查接口。
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query
import logging

from api.dependencies import get_analytics_service, get_current_user_id
from api.schemas.common import DataResponse, HealthResponse
from services import AnalyticsService

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()


@router.get("/dashboard", summary="仪表盘数据")
async def get_dashboard(
    user_id: str = Depends(get_current_user_id),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    获取仪表盘摘要数据
    
    包含：
    - 总记忆数
    - 今日新增
    - 文档数
    - 实体数
    - 记忆类型分布
    - 存储状态
    - 最近活动
    """
    summary = service.get_dashboard_summary(user_id=user_id)
    return DataResponse(success=True, data=summary.model_dump())


@router.get("/trends", summary="趋势报告")
async def get_trends(
    period: str = Query("week", description="周期: day/week/month"),
    user_id: str = Depends(get_current_user_id),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    获取趋势报告
    
    包含：
    - 记忆增长曲线
    - 平均重要性变化
    - 类型分布趋势
    """
    # 参数验证
    if period not in ("day", "week", "month"):
        period = "week"
    
    report = service.get_trend_report(period=period, user_id=user_id)
    return DataResponse(success=True, data=report.model_dump())


@router.get("/health", response_model=HealthResponse, summary="系统健康")
async def get_system_health(
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    获取系统健康状态
    
    检查各组件的连接和运行状态：
    - Qdrant 向量数据库
    - Neo4j 图数据库
    - SQLite 本地数据库
    - 嵌入模型
    """
    health = service.get_system_health()
    return HealthResponse(
        status=health.overall_status,
        components=health.components,
        timestamp=health.last_check
    )


@router.get("/query-stats", summary="查询统计")
async def get_query_stats(
    days: int = Query(7, ge=1, le=30, description="统计天数"),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    获取每日查询统计
    
    返回指定天数内每天的查询次数。
    """
    stats = service.get_query_stats(days=days)
    return DataResponse(success=True, data=stats)


@router.post("/log-activity", summary="记录活动")
async def log_activity(
    action: str = Query(..., description="动作类型"),
    details: Optional[dict] = None,
    user_id: str = Depends(get_current_user_id),
    service: AnalyticsService = Depends(get_analytics_service)
):
    """
    记录用户活动
    
    用于追踪用户行为和系统使用情况。
    """
    service.log_activity(action=action, user_id=user_id, details=details)
    return DataResponse(success=True, message="活动已记录")

