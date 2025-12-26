# 阶段0：项目准备 - 详细任务规划

## 阶段概述

| 属性 | 说明 |
|------|------|
| **阶段名称** | 项目基础设施搭建 |
| **预计时间** | 1天 |
| **核心目标** | 完成项目目录结构重组、依赖配置、配置管理系统和日志系统 |
| **前置条件** | 现有代码库可用，Python 3.10+ 环境 |

---

## Task 0.1：项目目录结构创建

### 任务描述
创建新的分层架构目录结构，为后续代码迁移做准备。

### 具体操作

#### 0.1.1 需要创建的目录结构
```
agents_memory/
├── api/                          # API层
│   ├── __init__.py
│   ├── routes/
│   │   └── __init__.py
│   ├── schemas/
│   │   └── __init__.py
│   └── middleware/
│       └── __init__.py
├── core/                         # 核心业务层
│   ├── __init__.py
│   ├── memory/
│   │   └── __init__.py
│   ├── rag/
│   │   └── __init__.py
│   ├── embedding/
│   │   └── __init__.py
│   └── storage/
│       └── __init__.py
├── services/                     # 业务服务层
│   └── __init__.py
├── web/                          # 前端资源
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   ├── templates/
│   └── components/
├── config/                       # 配置模块
│   └── __init__.py
├── utils/                        # 工具模块
│   └── __init__.py
├── tests/                        # 测试模块
│   └── __init__.py
├── scripts/                      # 脚本
├── logs/                         # 日志目录
└── data/                         # 数据目录
```

#### 0.1.2 各 `__init__.py` 文件内容

```python
# api/__init__.py
"""API层模块"""

# api/routes/__init__.py
"""API路由模块"""

# api/schemas/__init__.py
"""Pydantic数据模型"""

# api/middleware/__init__.py
"""中间件模块"""

# core/__init__.py
"""核心业务层"""

# core/memory/__init__.py
"""记忆系统核心模块"""

# core/rag/__init__.py
"""RAG检索增强生成模块"""

# core/embedding/__init__.py
"""嵌入服务模块"""

# core/storage/__init__.py
"""存储后端模块"""

# services/__init__.py
"""业务服务层"""

# config/__init__.py
"""配置管理模块"""

# utils/__init__.py
"""工具函数模块"""

# tests/__init__.py
"""测试模块"""
```

### 验证方法
```powershell
# Windows PowerShell 验证命令
Test-Path api, core, services, web, config, utils, tests, scripts, logs, data
Test-Path api/__init__.py, core/__init__.py, services/__init__.py, config/__init__.py, utils/__init__.py
```

### 预期结果
- 所有目录创建成功
- 所有 `__init__.py` 文件存在
- 目录结构清晰可用

---

## Task 0.2：依赖管理配置

### 任务描述
创建完整的 Python 项目依赖管理文件。

### 具体文件

#### 0.2.1 requirements.txt

```
# ===========================================
# Web框架与服务器
# ===========================================
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
jinja2>=3.1.2
aiofiles>=23.2.1

# ===========================================
# 数据验证与序列化
# ===========================================
pydantic>=2.5.0
pydantic-settings>=2.1.0

# ===========================================
# 数据库客户端
# ===========================================
qdrant-client>=1.6.0
neo4j>=5.0.0

# ===========================================
# 嵌入模型
# ===========================================
sentence-transformers>=2.2.0
dashscope>=1.14.0

# ===========================================
# 文档处理
# ===========================================
markitdown>=0.0.1a
pypdf>=3.17.0

# ===========================================
# NLP工具
# ===========================================
spacy>=3.7.0
langdetect>=1.0.9
tiktoken>=0.5.0

# ===========================================
# 数值计算与机器学习
# ===========================================
numpy>=1.24.0
scikit-learn>=1.3.0

# ===========================================
# 日志与工具
# ===========================================
loguru>=0.7.0
python-dotenv>=1.0.0
httpx>=0.25.0
tenacity>=8.2.0

# ===========================================
# 开发与测试
# ===========================================
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

#### 0.2.2 pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "agents-memory"
version = "1.0.0"
description = "智能体记忆与检索系统 - 可视化管理平台"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "HelloAgents Team"}
]
keywords = ["agent", "memory", "rag", "knowledge-graph", "llm"]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "isort>=5.12.0",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["api*", "core*", "services*", "config*", "utils*"]

[tool.black]
line-length = 100
target-version = ["py310", "py311", "py312"]

[tool.isort]
profile = "black"
line_length = 100
```

### 验证方法
```powershell
# 验证文件存在
Test-Path requirements.txt, pyproject.toml

# 验证依赖可解析（不实际安装）
pip install -r requirements.txt --dry-run
```

### 预期结果
- `requirements.txt` 包含所有必要依赖
- `pyproject.toml` 项目元数据正确

---

## Task 0.3：配置管理系统实现

### 任务描述
实现基于 Pydantic Settings 的配置管理系统，支持环境变量和 `.env` 文件。

### 具体实现

#### 0.3.1 config/settings.py

**类设计：**

| 类名 | 职责 | 主要属性 |
|------|------|----------|
| `AppSettings` | 应用基础配置 | `app_name`, `app_env`, `debug`, `host`, `port` |
| `DatabaseSettings` | 数据库配置 | `qdrant_url`, `neo4j_uri`, `sqlite_path` |
| `EmbeddingSettings` | 嵌入模型配置 | `embed_model_type`, `embed_model_name`, `embed_dimension` |
| `LLMSettings` | 大语言模型配置 | `llm_model_type`, `llm_model_name`, `llm_api_base` |
| `Settings` | 统一配置入口 | 聚合以上所有配置 |

**函数设计：**

| 函数名 | 功能 | 返回值 |
|--------|------|--------|
| `get_settings()` | 获取配置单例（带缓存） | `Settings` |

**代码实现：**

```python
"""
文件路径: config/settings.py
功能: 应用配置管理
"""

from functools import lru_cache
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """应用基础配置"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    app_name: str = Field(default="AgentMemorySystem")
    app_env: Literal["development", "production", "testing"] = Field(default="development")
    debug: bool = Field(default=True)
    secret_key: str = Field(default="change-me-in-production")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)


class DatabaseSettings(BaseSettings):
    """数据库配置"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: Optional[str] = Field(default=None)
    qdrant_collection: str = Field(default="hello_agents_vectors")
    
    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_username: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")
    
    # SQLite
    sqlite_path: str = Field(default="data/memory.db")


class EmbeddingSettings(BaseSettings):
    """嵌入模型配置"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    embed_model_type: Literal["dashscope", "local", "tfidf"] = Field(default="dashscope")
    embed_model_name: str = Field(default="text-embedding-v3")
    embed_api_key: Optional[str] = Field(default=None, alias="DASHSCOPE_API_KEY")
    embed_dimension: int = Field(default=1024)
    embed_batch_size: int = Field(default=25)


class LLMSettings(BaseSettings):
    """大语言模型配置（参考DashScope兼容模式）"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    llm_model_type: Literal["dashscope", "openai"] = Field(default="dashscope")
    llm_model_name: str = Field(default="qwen-plus")
    llm_api_key: Optional[str] = Field(default=None, alias="DASHSCOPE_API_KEY")
    llm_api_base: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    llm_max_tokens: int = Field(default=2048)
    llm_temperature: float = Field(default=0.7)


class Settings(BaseSettings):
    """统一配置入口"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


# 便捷访问
settings = get_settings()
```

#### 0.3.2 config/__init__.py

```python
"""配置模块导出"""

from .settings import (
    Settings, AppSettings, DatabaseSettings, 
    EmbeddingSettings, LLMSettings,
    get_settings, settings
)

__all__ = [
    "Settings", "AppSettings", "DatabaseSettings",
    "EmbeddingSettings", "LLMSettings", 
    "get_settings", "settings"
]
```

#### 0.3.3 .env.example

```bash
# 应用配置
APP_NAME=AgentMemorySystem
APP_ENV=development
DEBUG=true
SECRET_KEY=your-secret-key
HOST=0.0.0.0
PORT=8000

# Qdrant向量数据库
QDRANT_URL=https://your-cluster.qdrant.tech:6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=hello_agents_vectors

# Neo4j图数据库
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# SQLite
SQLITE_PATH=data/memory.db

# 嵌入模型
EMBED_MODEL_TYPE=dashscope
EMBED_MODEL_NAME=text-embedding-v3
EMBED_DIMENSION=1024

# LLM配置 (DashScope兼容OpenAI格式)
LLM_MODEL_TYPE=dashscope
LLM_MODEL_NAME=qwen-plus
LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.7

# 阿里云百炼API密钥
DASHSCOPE_API_KEY=your_dashscope_api_key
```

### 验证方法

**验证脚本：scripts/verify_config.py**

```python
"""配置系统验证脚本"""
import sys
sys.path.insert(0, '.')

def verify():
    from config import settings
    
    print("=" * 50)
    print("配置系统验证")
    print("=" * 50)
    
    # 验证各配置项
    checks = [
        ("应用名称", settings.app.app_name, bool),
        ("运行环境", settings.app.app_env, lambda x: x in ["development", "production", "testing"]),
        ("Qdrant URL", settings.database.qdrant_url, bool),
        ("Neo4j URI", settings.database.neo4j_uri, bool),
        ("嵌入模型类型", settings.embedding.embed_model_type, lambda x: x in ["dashscope", "local", "tfidf"]),
        ("LLM API Base", settings.llm.llm_api_base, lambda x: x.startswith("http")),
    ]
    
    all_passed = True
    for name, value, validator in checks:
        try:
            passed = validator(value) if callable(validator) else bool(value)
            status = "✓" if passed else "✗"
            print(f"{status} {name}: {value}")
            all_passed = all_passed and passed
        except Exception as e:
            print(f"✗ {name}: 验证失败 - {e}")
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("所有配置验证通过！")
    else:
        print("部分配置验证失败！")
    
    return all_passed

if __name__ == "__main__":
    verify()
```

**运行验证：**
```powershell
python scripts/verify_config.py
```

### 预期结果
```
==================================================
配置系统验证
==================================================
✓ 应用名称: AgentMemorySystem
✓ 运行环境: development
✓ Qdrant URL: http://localhost:6333
✓ Neo4j URI: bolt://localhost:7687
✓ 嵌入模型类型: dashscope
✓ LLM API Base: https://dashscope.aliyuncs.com/compatible-mode/v1
==================================================
所有配置验证通过！
```

---

## Task 0.4：日志系统实现

### 任务描述
基于 Loguru 实现结构化日志系统，支持控制台彩色输出和文件记录。

### 具体实现

#### 0.4.1 config/logging.py

**函数设计：**

| 函数名 | 功能 | 参数 | 返回值 |
|--------|------|------|--------|
| `setup_logging()` | 初始化日志系统 | 无 | `None` |
| `get_logger(name)` | 获取命名日志器 | `name: str` | `logger` |

**代码实现：**

```python
"""
文件路径: config/logging.py
功能: 日志系统配置
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(debug: bool = True, log_dir: str = "logs") -> None:
    """配置日志系统
    
    Args:
        debug: 是否开启调试模式
        log_dir: 日志目录路径
    """
    # 移除默认处理器
    logger.remove()
    
    # 日志格式
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{function}:{line} | {message}"
    )
    
    # 控制台输出
    logger.add(
        sys.stderr,
        format=console_format,
        level="DEBUG" if debug else "INFO",
        colorize=True,
        backtrace=True,
        diagnose=debug
    )
    
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # 普通日志文件
    logger.add(
        log_path / "app_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="INFO",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        encoding="utf-8"
    )
    
    # 错误日志文件
    logger.add(
        log_path / "error_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="ERROR",
        rotation="00:00",
        retention="90 days",
        compression="zip",
        encoding="utf-8"
    )
    
    logger.info("日志系统初始化完成")


def get_logger(name: str = None):
    """获取命名日志器
    
    Args:
        name: 日志器名称，通常使用 __name__
    
    Returns:
        绑定名称的日志器
    """
    if name:
        return logger.bind(name=name)
    return logger
```

#### 0.4.2 更新 config/__init__.py

```python
"""配置模块导出"""

from .settings import (
    Settings, AppSettings, DatabaseSettings, 
    EmbeddingSettings, LLMSettings,
    get_settings, settings
)
from .logging import setup_logging, get_logger

__all__ = [
    "Settings", "AppSettings", "DatabaseSettings",
    "EmbeddingSettings", "LLMSettings", 
    "get_settings", "settings",
    "setup_logging", "get_logger"
]
```

### 验证方法

**验证脚本：scripts/verify_logging.py**

```python
"""日志系统验证脚本"""
import sys
from pathlib import Path
sys.path.insert(0, '.')

def verify():
    from config import setup_logging, get_logger
    
    print("=" * 50)
    print("日志系统验证")
    print("=" * 50)
    
    # 初始化日志
    setup_logging(debug=True)
    
    # 获取日志器
    log = get_logger("verify_test")
    
    # 测试各级别
    log.debug("DEBUG级别日志")
    log.info("INFO级别日志")
    log.warning("WARNING级别日志")
    log.error("ERROR级别日志")
    
    # 测试异常
    try:
        raise ValueError("测试异常")
    except Exception as e:
        log.exception(f"异常捕获: {e}")
    
    # 验证日志文件
    log_dir = Path("logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        print(f"\n✓ 日志目录存在: {log_dir}")
        print(f"✓ 日志文件数量: {len(log_files)}")
        for f in log_files:
            print(f"  - {f.name}")
        return True
    else:
        print("✗ 日志目录不存在")
        return False

if __name__ == "__main__":
    verify()
```

**运行验证：**
```powershell
python scripts/verify_logging.py
```

### 预期结果
- 控制台显示彩色日志输出
- `logs/` 目录创建成功
- 日志文件 `app_YYYY-MM-DD.log` 和 `error_YYYY-MM-DD.log` 生成

---

## Task 0.5：工具函数模块实现

### 任务描述
创建通用工具函数模块，提供ID生成、时间处理、文本处理等辅助功能。

### 具体实现

#### 0.5.1 utils/helpers.py

**函数设计：**

| 函数名 | 功能 | 参数 | 返回值 |
|--------|------|------|--------|
| `generate_id(prefix)` | 生成唯一ID | `prefix: str = ""` | `str` |
| `generate_hash(content)` | 生成SHA256哈希 | `content: str` | `str` |
| `timestamp_now()` | 获取ISO时间戳 | 无 | `str` |
| `truncate_text(text, max_len)` | 截断文本 | `text: str, max_len: int` | `str` |
| `chunks(lst, n)` | 列表分块 | `lst: List, n: int` | `Generator` |

**代码实现：**

```python
"""
文件路径: utils/helpers.py
功能: 通用辅助函数
"""

import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict, List, Generator


def generate_id(prefix: str = "") -> str:
    """生成唯一ID
    
    Args:
        prefix: ID前缀
    
    Returns:
        格式: {prefix}_{uuid} 或 {uuid}
    """
    uid = str(uuid.uuid4())
    return f"{prefix}_{uid}" if prefix else uid


def generate_hash(content: str) -> str:
    """生成内容SHA256哈希
    
    Args:
        content: 要哈希的内容
    
    Returns:
        64位十六进制哈希字符串
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def timestamp_now() -> str:
    """获取当前ISO格式时间戳
    
    Returns:
        ISO格式时间字符串，如 2024-12-26T10:30:00.123456
    """
    return datetime.now().isoformat()


def timestamp_to_datetime(ts: str) -> datetime:
    """ISO时间戳转datetime对象
    
    Args:
        ts: ISO格式时间字符串
    
    Returns:
        datetime对象
    """
    return datetime.fromisoformat(ts)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断文本到指定长度
    
    Args:
        text: 原始文本
        max_length: 最大长度（包含后缀）
        suffix: 截断后缀
    
    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def chunks(lst: List[Any], n: int) -> Generator[List[Any], None, None]:
    """将列表分成大小为n的块
    
    Args:
        lst: 原始列表
        n: 每块大小
    
    Yields:
        子列表
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def safe_get(data: Dict, key: str, default: Any = None) -> Any:
    """安全获取嵌套字典值
    
    Args:
        data: 字典
        key: 点分隔的键路径，如 "a.b.c"
        default: 默认值
    
    Returns:
        对应值或默认值
    """
    keys = key.split('.')
    result = data
    for k in keys:
        if isinstance(result, dict):
            result = result.get(k)
        else:
            return default
        if result is None:
            return default
    return result
```

#### 0.5.2 utils/__init__.py

```python
"""工具模块导出"""

from .helpers import (
    generate_id,
    generate_hash,
    timestamp_now,
    timestamp_to_datetime,
    truncate_text,
    chunks,
    safe_get
)

__all__ = [
    "generate_id",
    "generate_hash",
    "timestamp_now",
    "timestamp_to_datetime",
    "truncate_text",
    "chunks",
    "safe_get"
]
```

### 验证方法

**验证脚本：scripts/verify_utils.py**

```python
"""工具函数验证脚本"""
import sys
sys.path.insert(0, '.')

def verify():
    from utils import (
        generate_id, generate_hash, timestamp_now,
        truncate_text, chunks, safe_get
    )
    
    print("=" * 50)
    print("工具函数验证")
    print("=" * 50)
    
    all_passed = True
    
    # 测试 generate_id
    id1 = generate_id("mem")
    id2 = generate_id()
    passed = id1.startswith("mem_") and id1 != id2
    print(f"{'✓' if passed else '✗'} generate_id: {id1[:20]}...")
    all_passed = all_passed and passed
    
    # 测试 generate_hash
    h1 = generate_hash("hello")
    h2 = generate_hash("hello")
    passed = h1 == h2 and len(h1) == 64
    print(f"{'✓' if passed else '✗'} generate_hash: {h1[:16]}...")
    all_passed = all_passed and passed
    
    # 测试 timestamp_now
    ts = timestamp_now()
    passed = "T" in ts and len(ts) > 10
    print(f"{'✓' if passed else '✗'} timestamp_now: {ts}")
    all_passed = all_passed and passed
    
    # 测试 truncate_text
    text = "这是一段很长的测试文本" * 5
    truncated = truncate_text(text, 20)
    passed = len(truncated) == 20 and truncated.endswith("...")
    print(f"{'✓' if passed else '✗'} truncate_text: {truncated}")
    all_passed = all_passed and passed
    
    # 测试 chunks
    lst = list(range(10))
    chunked = list(chunks(lst, 3))
    passed = len(chunked) == 4 and chunked[0] == [0, 1, 2]
    print(f"{'✓' if passed else '✗'} chunks: {chunked}")
    all_passed = all_passed and passed
    
    # 测试 safe_get
    data = {"a": {"b": {"c": 42}}}
    passed = safe_get(data, "a.b.c") == 42 and safe_get(data, "x.y", "default") == "default"
    print(f"{'✓' if passed else '✗'} safe_get: a.b.c={safe_get(data, 'a.b.c')}")
    all_passed = all_passed and passed
    
    print("=" * 50)
    if all_passed:
        print("所有工具函数验证通过！")
    else:
        print("部分工具函数验证失败！")
    
    return all_passed

if __name__ == "__main__":
    verify()
```

**运行验证：**
```powershell
python scripts/verify_utils.py
```

### 预期结果
```
==================================================
工具函数验证
==================================================
✓ generate_id: mem_xxxxxxxx-xxxx...
✓ generate_hash: 2cf24dba5fb0a30e...
✓ timestamp_now: 2024-12-26T10:30:00.123456
✓ truncate_text: 这是一段很长的测试文本这是...
✓ chunks: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
✓ safe_get: a.b.c=42
==================================================
所有工具函数验证通过！
```

---

## 阶段0 总验证清单

| 序号 | 任务 | 验证命令 | 预期结果 |
|------|------|----------|----------|
| 1 | 目录结构 | `Test-Path api,core,services,web,config,utils` | 全部返回 True |
| 2 | 依赖文件 | `Test-Path requirements.txt,pyproject.toml,.env.example` | 全部返回 True |
| 3 | 配置系统 | `python scripts/verify_config.py` | 输出"所有配置验证通过" |
| 4 | 日志系统 | `python scripts/verify_logging.py` | 日志文件生成，控制台输出正确 |
| 5 | 工具函数 | `python scripts/verify_utils.py` | 输出"所有工具函数验证通过" |
| 6 | 模块导入 | `python -c "from config import settings; from utils import generate_id; print('OK')"` | 输出 OK |

---

## 产出物清单

| 文件/目录 | 类型 | 说明 |
|-----------|------|------|
| `api/`, `core/`, `services/`, `web/`, `config/`, `utils/`, `tests/`, `scripts/`, `logs/`, `data/` | 目录 | 项目结构 |
| `requirements.txt` | 配置 | Python依赖 |
| `pyproject.toml` | 配置 | 项目元数据 |
| `.env.example` | 配置 | 环境变量示例 |
| `config/settings.py` | Python | 配置管理类 |
| `config/logging.py` | Python | 日志系统 |
| `utils/helpers.py` | Python | 工具函数 |
| `scripts/verify_config.py` | Python | 配置验证脚本 |
| `scripts/verify_logging.py` | Python | 日志验证脚本 |
| `scripts/verify_utils.py` | Python | 工具验证脚本 |

---

## 下一阶段预告

**阶段1：核心层重构** 将完成：
- 嵌入模块重构（`core/embedding/`）
- 存储模块重构（`core/storage/`）
- 记忆模块迁移（`core/memory/`）
- RAG模块迁移（`core/rag/`）

