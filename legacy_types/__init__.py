"""记忆类型层模块

为避免与Python标准库types模块冲突，
重新导出标准库内容并添加自定义类型。
"""

# 首先重新导出Python标准库types模块的所有内容
# 这是必需的，因为这个目录名与标准库冲突
import sys as _sys
import importlib as _importlib

# 从标准库导入真正的types模块
_stdlib_types = _importlib.import_module('types', package=None)

# 确保我们有正确的标准库types
if hasattr(_stdlib_types, 'GenericAlias'):
    from types import GenericAlias
if hasattr(_stdlib_types, 'ModuleType'):
    from types import ModuleType
if hasattr(_stdlib_types, 'FunctionType'):
    from types import FunctionType
if hasattr(_stdlib_types, 'MethodType'):
    from types import MethodType
if hasattr(_stdlib_types, 'LambdaType'):
    from types import LambdaType
if hasattr(_stdlib_types, 'GeneratorType'):
    from types import GeneratorType
if hasattr(_stdlib_types, 'CoroutineType'):
    from types import CoroutineType
if hasattr(_stdlib_types, 'AsyncGeneratorType'):
    from types import AsyncGeneratorType
if hasattr(_stdlib_types, 'CodeType'):
    from types import CodeType
if hasattr(_stdlib_types, 'CellType'):
    from types import CellType
if hasattr(_stdlib_types, 'FrameType'):
    from types import FrameType
if hasattr(_stdlib_types, 'TracebackType'):
    from types import TracebackType
if hasattr(_stdlib_types, 'BuiltinFunctionType'):
    from types import BuiltinFunctionType
if hasattr(_stdlib_types, 'BuiltinMethodType'):
    from types import BuiltinMethodType
if hasattr(_stdlib_types, 'WrapperDescriptorType'):
    from types import WrapperDescriptorType
if hasattr(_stdlib_types, 'MethodWrapperType'):
    from types import MethodWrapperType
if hasattr(_stdlib_types, 'MethodDescriptorType'):
    from types import MethodDescriptorType
if hasattr(_stdlib_types, 'ClassMethodDescriptorType'):
    from types import ClassMethodDescriptorType
if hasattr(_stdlib_types, 'MappingProxyType'):
    from types import MappingProxyType
if hasattr(_stdlib_types, 'SimpleNamespace'):
    from types import SimpleNamespace
if hasattr(_stdlib_types, 'DynamicClassAttribute'):
    from types import DynamicClassAttribute
if hasattr(_stdlib_types, 'UnionType'):
    from types import UnionType
if hasattr(_stdlib_types, 'NoneType'):
    from types import NoneType
if hasattr(_stdlib_types, 'EllipsisType'):
    from types import EllipsisType
if hasattr(_stdlib_types, 'NotImplementedType'):
    from types import NotImplementedType
if hasattr(_stdlib_types, 'new_class'):
    from types import new_class
if hasattr(_stdlib_types, 'resolve_bases'):
    from types import resolve_bases
if hasattr(_stdlib_types, 'prepare_class'):
    from types import prepare_class
if hasattr(_stdlib_types, 'get_original_bases'):
    from types import get_original_bases
if hasattr(_stdlib_types, 'coroutine'):
    from types import coroutine


# 延迟导入项目的记忆类型（向后兼容）
def __getattr__(name):
    """延迟加载记忆类型"""
    if name == "WorkingMemory":
        from core.memory.working import WorkingMemory
        return WorkingMemory
    elif name == "EpisodicMemory":
        from core.memory.episodic import EpisodicMemory
        return EpisodicMemory
    elif name == "SemanticMemory":
        from core.memory.semantic import SemanticMemory
        return SemanticMemory
    elif name == "PerceptualMemory":
        from core.memory.perceptual import PerceptualMemory
        return PerceptualMemory
    elif name == "Episode":
        from core.memory.episodic import Episode
        return Episode
    elif name == "Perception":
        from core.memory.perceptual import Perception
        return Perception
    
    # 尝试从标准库types获取
    if hasattr(_stdlib_types, name):
        return getattr(_stdlib_types, name)
    
    raise AttributeError(f"module 'types' has no attribute '{name}'")
