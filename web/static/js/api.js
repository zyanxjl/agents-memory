/**
 * API 调用封装
 * 提供统一的 API 调用接口
 */

const API_BASE = '/api/v1';

/**
 * 通用请求方法
 * @param {string} endpoint - API端点
 * @param {object} options - 请求选项
 * @returns {Promise} 响应数据
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    const config = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(url, config);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || data.message || '请求失败');
        }
        
        return data;
    } catch (error) {
        console.error(`API Error [${endpoint}]:`, error);
        throw error;
    }
}

// ==================== 记忆 API ====================

const MemoryAPI = {
    // 获取记忆列表
    list: (params = {}) => {
        const query = new URLSearchParams(params).toString();
        return apiRequest(`/memory?${query}`);
    },
    
    // 获取单个记忆
    get: (id) => apiRequest(`/memory/${id}`),
    
    // 添加记忆
    create: (data) => apiRequest('/memory', {
        method: 'POST',
        body: JSON.stringify(data),
    }),
    
    // 更新记忆
    update: (id, data) => apiRequest(`/memory/${id}`, {
        method: 'PUT',
        body: JSON.stringify(data),
    }),
    
    // 删除记忆
    delete: (id) => apiRequest(`/memory/${id}`, {
        method: 'DELETE',
    }),
    
    // 搜索记忆
    search: (data) => apiRequest('/memory/search', {
        method: 'POST',
        body: JSON.stringify(data),
    }),
    
    // 获取统计
    stats: () => apiRequest('/memory/stats/overview'),
    
    // 整合记忆
    consolidate: (data) => apiRequest('/memory/consolidate', {
        method: 'POST',
        body: JSON.stringify(data),
    }),
    
    // 遗忘记忆
    forget: (data) => apiRequest('/memory/forget', {
        method: 'POST',
        body: JSON.stringify(data),
    }),
};

// ==================== RAG API ====================

const RAGAPI = {
    // 上传文档
    uploadDocument: async (file, options = {}) => {
        const formData = new FormData();
        formData.append('file', file);
        
        const params = new URLSearchParams(options).toString();
        const response = await fetch(`${API_BASE}/rag/documents?${params}`, {
            method: 'POST',
            body: formData,
        });
        return response.json();
    },
    
    // 获取文档列表
    listDocuments: (params = {}) => {
        const query = new URLSearchParams(params).toString();
        return apiRequest(`/rag/documents?${query}`);
    },
    
    // 删除文档
    deleteDocument: (docId) => apiRequest(`/rag/documents/${docId}`, {
        method: 'DELETE',
    }),
    
    // 知识检索
    search: (data) => apiRequest('/rag/search', {
        method: 'POST',
        body: JSON.stringify(data),
    }),
    
    // 知识问答
    ask: (data) => apiRequest('/rag/ask', {
        method: 'POST',
        body: JSON.stringify(data),
    }),
    
    // 获取统计
    stats: () => apiRequest('/rag/stats'),
};

// ==================== 图谱 API ====================

const GraphAPI = {
    // 搜索实体
    searchEntities: (data) => apiRequest('/graph/entities/search', {
        method: 'POST',
        body: JSON.stringify(data),
    }),
    
    // 获取可视化数据
    visualization: (params = {}) => {
        const query = new URLSearchParams(params).toString();
        return apiRequest(`/graph/visualization?${query}`);
    },
    
    // 获取相关实体
    relatedEntities: (entityId, params = {}) => {
        const query = new URLSearchParams(params).toString();
        return apiRequest(`/graph/entities/${entityId}/related?${query}`);
    },
    
    // 获取统计
    stats: () => apiRequest('/graph/stats'),
};

// ==================== 分析 API ====================

const AnalyticsAPI = {
    // 获取仪表盘数据
    dashboard: () => apiRequest('/analytics/dashboard'),
    
    // 获取趋势报告
    trends: (period = 'week') => apiRequest(`/analytics/trends?period=${period}`),
    
    // 获取系统健康状态
    health: () => apiRequest('/analytics/health'),
    
    // 记录活动
    logActivity: (action, details = {}) => apiRequest(`/analytics/log-activity?action=${action}`, {
        method: 'POST',
        body: JSON.stringify(details),
    }),
};

// 导出到全局
window.MemoryAPI = MemoryAPI;
window.RAGAPI = RAGAPI;
window.GraphAPI = GraphAPI;
window.AnalyticsAPI = AnalyticsAPI;

