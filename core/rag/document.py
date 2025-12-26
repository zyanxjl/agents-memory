"""
文件路径: core/rag/document.py
功能: 文档处理和分块

提供:
- MarkItDown文档转换
- 智能分块策略
- 多语言支持
"""

from typing import List, Dict, Optional
import os
import hashlib


def _get_markitdown_instance():
    """获取MarkItDown实例"""
    try:
        from markitdown import MarkItDown
        return MarkItDown()
    except ImportError:
        print("[WARNING] MarkItDown不可用。请安装: pip install markitdown")
        return None


def _is_markitdown_supported_format(path: str) -> bool:
    """检查文件格式是否被MarkItDown支持"""
    ext = (os.path.splitext(path)[1] or '').lower()
    supported_formats = {
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp',
        '.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg',
        '.zip', '.tar', '.gz', '.rar',
        '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.css', '.scss',
        '.log', '.conf', '.ini', '.cfg', '.yaml', '.yml', '.toml'
    }
    return ext in supported_formats


def _convert_to_markdown(path: str) -> str:
    """将文件转换为Markdown文本"""
    if not os.path.exists(path):
        return ""
    
    ext = (os.path.splitext(path)[1] or '').lower()
    if ext == '.pdf':
        return _enhanced_pdf_processing(path)
    
    md_instance = _get_markitdown_instance()
    if md_instance is None:
        return _fallback_text_reader(path)
    
    try:
        result = md_instance.convert(path)
        text = getattr(result, "text_content", None)
        if isinstance(text, str) and text.strip():
            return text
        return ""
    except Exception as e:
        print(f"[WARNING] MarkItDown处理失败 {path}: {e}")
        return _fallback_text_reader(path)


def _enhanced_pdf_processing(path: str) -> str:
    """增强的PDF处理"""
    print(f"[RAG] 使用增强PDF处理: {path}")
    
    md_instance = _get_markitdown_instance()
    if md_instance is None:
        return _fallback_text_reader(path)
    
    try:
        result = md_instance.convert(path)
        raw_text = getattr(result, "text_content", None)
        if not raw_text or not raw_text.strip():
            return ""
        
        cleaned_text = _post_process_pdf_text(raw_text)
        print(f"[RAG] PDF后处理完成: {len(raw_text)} -> {len(cleaned_text)} 字符")
        return cleaned_text
        
    except Exception as e:
        print(f"[WARNING] 增强PDF处理失败 {path}: {e}")
        return _fallback_text_reader(path)


def _post_process_pdf_text(text: str) -> str:
    """PDF文本后处理"""
    import re
    
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) <= 2 and not line.isdigit():
            continue
        if re.match(r'^\d+$', line):
            continue
        cleaned_lines.append(line)
    
    # 智能合并短行
    merged_lines = []
    i = 0
    
    while i < len(cleaned_lines):
        current_line = cleaned_lines[i]
        
        if len(current_line) < 60 and i + 1 < len(cleaned_lines):
            next_line = cleaned_lines[i + 1]
            
            if (not current_line.endswith('：') and 
                not current_line.endswith(':') and
                not current_line.startswith('#') and
                not next_line.startswith('#') and
                len(next_line) < 120):
                
                merged_line = current_line + " " + next_line
                merged_lines.append(merged_line)
                i += 2
                continue
        
        merged_lines.append(current_line)
        i += 1
    
    return '\n\n'.join(merged_lines)


def _fallback_text_reader(path: str) -> str:
    """回退文本读取器"""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        try:
            with open(path, 'r', encoding='latin-1', errors='ignore') as f:
                return f.read()
        except Exception:
            return ""


def _detect_lang(sample: str) -> str:
    """检测语言"""
    try:
        from langdetect import detect
        return detect(sample[:1000]) if sample else "unknown"
    except Exception:
        return "unknown"


def _is_cjk(ch: str) -> bool:
    """判断是否为CJK字符"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF or
        0x3400 <= code <= 0x4DBF or
        0x20000 <= code <= 0x2A6DF
    )


def _approx_token_len(text: str) -> int:
    """近似估计token长度"""
    cjk = sum(1 for ch in text if _is_cjk(ch))
    non_cjk_tokens = len([t for t in text.split() if t])
    return cjk + non_cjk_tokens


def _split_paragraphs_with_headings(text: str) -> List[Dict]:
    """按标题分割段落"""
    lines = text.splitlines()
    heading_stack: List[str] = []
    paragraphs: List[Dict] = []
    buf: List[str] = []
    char_pos = 0
    
    def flush_buf(end_pos: int):
        if not buf:
            return
        content = "\n".join(buf).strip()
        if not content:
            return
        paragraphs.append({
            "content": content,
            "heading_path": " > ".join(heading_stack) if heading_stack else None,
            "start": max(0, end_pos - len(content)),
            "end": end_pos,
        })
    
    for ln in lines:
        raw = ln
        if raw.strip().startswith("#"):
            flush_buf(char_pos)
            level = len(raw) - len(raw.lstrip('#'))
            title = raw.lstrip('#').strip()
            if level <= 0:
                level = 1
            if level <= len(heading_stack):
                heading_stack = heading_stack[:level-1]
            heading_stack.append(title)
            char_pos += len(raw) + 1
            continue
        if raw.strip() == "":
            flush_buf(char_pos)
            buf = []
        else:
            buf.append(raw)
        char_pos += len(raw) + 1
    flush_buf(char_pos)
    
    if not paragraphs:
        paragraphs = [{"content": text, "heading_path": None, "start": 0, "end": len(text)}]
    return paragraphs


def _chunk_paragraphs(paragraphs: List[Dict], chunk_tokens: int, overlap_tokens: int) -> List[Dict]:
    """分块段落"""
    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0
    i = 0
    
    while i < len(paragraphs):
        p = paragraphs[i]
        p_tokens = _approx_token_len(p["content"]) or 1
        if cur_tokens + p_tokens <= chunk_tokens or not cur:
            cur.append(p)
            cur_tokens += p_tokens
            i += 1
        else:
            content = "\n\n".join(x["content"] for x in cur)
            start = cur[0]["start"]
            end = cur[-1]["end"]
            heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)
            chunks.append({
                "content": content,
                "start": start,
                "end": end,
                "heading_path": heading_path,
            })
            if overlap_tokens > 0 and cur:
                kept: List[Dict] = []
                kept_tokens = 0
                for x in reversed(cur):
                    t = _approx_token_len(x["content"]) or 1
                    if kept_tokens + t > overlap_tokens:
                        break
                    kept.append(x)
                    kept_tokens += t
                cur = list(reversed(kept))
                cur_tokens = kept_tokens
            else:
                cur = []
                cur_tokens = 0
    
    if cur:
        content = "\n\n".join(x["content"] for x in cur)
        start = cur[0]["start"]
        end = cur[-1]["end"]
        heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)
        chunks.append({
            "content": content,
            "start": start,
            "end": end,
            "heading_path": heading_path,
        })
    return chunks


def load_and_chunk_texts(
    paths: List[str], 
    chunk_size: int = 800, 
    chunk_overlap: int = 100, 
    namespace: Optional[str] = None, 
    source_label: str = "rag"
) -> List[Dict]:
    """通用文档加载和分块
    
    Args:
        paths: 文件路径列表
        chunk_size: 分块大小（token数）
        chunk_overlap: 分块重叠（token数）
        namespace: 命名空间
        source_label: 来源标签
        
    Returns:
        分块列表
    """
    print(f"[RAG] 开始加载: 文件数={len(paths)} 分块大小={chunk_size} 重叠={chunk_overlap}")
    chunks: List[Dict] = []
    seen_hashes = set()
    
    for path in paths:
        if not os.path.exists(path):
            print(f"[WARNING] 文件不存在: {path}")
            continue
            
        print(f"[RAG] 处理: {path}")
        ext = (os.path.splitext(path)[1] or '').lower()
        
        markdown_text = _convert_to_markdown(path)
        if not markdown_text.strip():
            print(f"[WARNING] 无法提取内容: {path}")
            continue
        
        lang = _detect_lang(markdown_text)
        doc_id = hashlib.md5(f"{path}|{len(markdown_text)}".encode('utf-8')).hexdigest()
        
        para = _split_paragraphs_with_headings(markdown_text)
        token_chunks = _chunk_paragraphs(para, chunk_tokens=max(1, chunk_size), overlap_tokens=max(0, chunk_overlap))
        
        for ch in token_chunks:
            content = ch["content"]
            start = ch.get("start", 0)
            end = ch.get("end", start + len(content))
            norm = content.strip()
            if not norm:
                continue
                
            content_hash = hashlib.md5(norm.encode('utf-8')).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            
            chunk_id = hashlib.md5(f"{doc_id}|{start}|{end}|{content_hash}".encode('utf-8')).hexdigest()
            chunks.append({
                "id": chunk_id,
                "content": content,
                "metadata": {
                    "source_path": path,
                    "file_ext": ext,
                    "doc_id": doc_id,
                    "lang": lang,
                    "start": start,
                    "end": end,
                    "content_hash": content_hash,
                    "namespace": namespace or "default",
                    "source": source_label,
                    "heading_path": ch.get("heading_path"),
                    "format": "markdown",
                },
            })
            
    print(f"[RAG] 加载完成: 总分块数={len(chunks)}")
    return chunks

