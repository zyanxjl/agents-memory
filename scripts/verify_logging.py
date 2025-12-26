"""日志系统验证脚本"""
import sys
from pathlib import Path

# 设置控制台编码
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

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
        print(f"\n[OK] 日志目录存在: {log_dir}")
        print(f"[OK] 日志文件数量: {len(log_files)}")
        for f in log_files:
            print(f"  - {f.name}")
        print("\n日志系统验证通过!")
        return True
    else:
        print("[FAIL] 日志目录不存在")
        return False

if __name__ == "__main__":
    verify()
