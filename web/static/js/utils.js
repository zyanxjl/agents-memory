/**
 * 工具函数集合
 * Agent Memory System
 */

/**
 * 显示通知
 * @param {string} message - 消息内容
 * @param {string} type - 类型: success/error/warning/info
 * @param {number} duration - 显示时长(毫秒)
 */
function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const toast = document.createElement('div');
    
    const colors = {
        success: 'bg-green-600',
        error: 'bg-red-600',
        warning: 'bg-yellow-600',
        info: 'bg-blue-600',
    };
    
    const icons = {
        success: `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>`,
        error: `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>`,
        warning: `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>`,
        info: `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`,
    };
    
    toast.className = `flex items-center space-x-3 px-4 py-3 rounded-lg shadow-lg ${colors[type]} text-white transform transition-all duration-300 translate-x-full`;
    toast.innerHTML = `${icons[type]}<span>${message}</span>`;
    
    container.appendChild(toast);
    
    // 动画显示
    requestAnimationFrame(() => {
        toast.classList.remove('translate-x-full');
    });
    
    // 自动隐藏
    setTimeout(() => {
        toast.classList.add('translate-x-full');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

/**
 * 格式化日期时间
 * @param {string|Date} date - 日期
 * @param {string} format - 格式
 * @returns {string}
 */
function formatDateTime(date, format = 'YYYY-MM-DD HH:mm') {
    const d = new Date(date);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hours = String(d.getHours()).padStart(2, '0');
    const minutes = String(d.getMinutes()).padStart(2, '0');
    const seconds = String(d.getSeconds()).padStart(2, '0');
    
    return format
        .replace('YYYY', year)
        .replace('MM', month)
        .replace('DD', day)
        .replace('HH', hours)
        .replace('mm', minutes)
        .replace('ss', seconds);
}

/**
 * 格式化相对时间
 * @param {string|Date} date - 日期
 * @returns {string}
 */
function formatRelativeTime(date) {
    const now = new Date();
    const d = new Date(date);
    const diff = now - d;
    
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    
    if (days > 7) return formatDateTime(date, 'MM-DD HH:mm');
    if (days > 0) return `${days}天前`;
    if (hours > 0) return `${hours}小时前`;
    if (minutes > 0) return `${minutes}分钟前`;
    return '刚刚';
}

/**
 * 格式化文件大小
 * @param {number} bytes - 字节数
 * @returns {string}
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * 截断文本
 * @param {string} text - 文本
 * @param {number} length - 最大长度
 * @returns {string}
 */
function truncateText(text, length = 100) {
    if (!text) return '';
    if (text.length <= length) return text;
    return text.substring(0, length) + '...';
}

/**
 * 获取记忆类型标签HTML
 * @param {string} type - 记忆类型
 * @returns {string}
 */
function getMemoryTypeBadge(type) {
    const styles = {
        working: 'memory-type-working',
        episodic: 'memory-type-episodic',
        semantic: 'memory-type-semantic',
        perceptual: 'memory-type-perceptual',
    };
    const labels = {
        working: '工作记忆',
        episodic: '情景记忆',
        semantic: '语义记忆',
        perceptual: '感知记忆',
    };
    return `<span class="badge ${styles[type] || 'badge-info'}">${labels[type] || type}</span>`;
}

/**
 * 复制到剪贴板
 * @param {string} text - 文本
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showToast('已复制到剪贴板', 'success');
    } catch (error) {
        showToast('复制失败', 'error');
    }
}

/**
 * 防抖函数
 * @param {Function} func - 要防抖的函数
 * @param {number} wait - 等待时间(毫秒)
 * @returns {Function}
 */
function debounce(func, wait = 300) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

/**
 * 确认对话框
 * @param {string} message - 消息
 * @returns {Promise<boolean>}
 */
function confirmDialog(message) {
    return new Promise((resolve) => {
        if (confirm(message)) {
            resolve(true);
        } else {
            resolve(false);
        }
    });
}

// 导出到全局
window.showToast = showToast;
window.formatDateTime = formatDateTime;
window.formatRelativeTime = formatRelativeTime;
window.formatFileSize = formatFileSize;
window.truncateText = truncateText;
window.getMemoryTypeBadge = getMemoryTypeBadge;
window.copyToClipboard = copyToClipboard;
window.debounce = debounce;
window.confirmDialog = confirmDialog;

