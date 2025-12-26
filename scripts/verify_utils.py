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

