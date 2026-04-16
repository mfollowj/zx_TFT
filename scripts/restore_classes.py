"""
restore_classes.py
独立脚本，不依赖 ultralytics。
每次用 labelimg 之前运行一次，把完整原始类同步回 pending_DO 目录。
"""
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils._encoding import read_lines, write_text

MASTER = r'C:\TFT_CHESS_DO\classes.txt'
TARGET = r'C:\TFT_CHESS_pending_DO\classes.txt'

if not os.path.exists(MASTER):
    print(f'[ERROR] 找不到原始类文件: {MASTER}')
    exit(1)

# 自动检测编码读取（兼容旧 GBK 文件）
classes = read_lines(MASTER)

# 统一写入 UTF-8
content = '\n'.join(c.strip() for c in classes if c.strip())
write_text(TARGET, content)

print(f'[OK] 已恢复 {len(classes)} 个类到 pending_DO')
for i, c in enumerate(classes):
    print(f'  {i}: {c}')
