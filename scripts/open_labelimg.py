"""
open_labelimg.py
打开 labelimg 前，先把 classes.txt 设为只读（防止 labelimg 截断），
然后启动 labelimg 指向 pending_DO 目录。
"""
import os, sys, subprocess, stat

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PENDING_DO = r'C:\TFT_CHESS_pending_DO'
CLASSES_FILE = os.path.join(PENDING_DO, 'classes.txt')

# 1. 确保 classes.txt 有完整内容（用 restore_classes 逻辑）
MASTER = r'C:\TFT_CHESS_DO\classes.txt'
if os.path.exists(MASTER):
    from utils._encoding import read_lines
    classes = read_lines(MASTER)
    with open(CLASSES_FILE, 'w', encoding='utf-8') as f:
        for c in classes:
            f.write(c + '\n')
    print(f'[OK] classes.txt 已恢复 {len(classes)} 个类')

# 2. 设为只读（labelimg 写不进去）
os.chmod(CLASSES_FILE, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
print(f'[OK] classes.txt 已设为只读 → {CLASSES_FILE}')

# 3. 启动 labelimg
print(f'\n启动 labelimg ...')
subprocess.Popen(
    ['labelImg', PENDING_DO],
    creationflags=subprocess.CREATE_NEW_CONSOLE
)

print('---')
print('labelimg 已打开，classes.txt 已保护。')
print('新增的标注类别请手动追加到 C:/TFT_CHESS_DO/classes.txt')
print('关闭 labelimg 后想恢复写入权限，运行:')
print('  attrib -R C:\\TFT_CHESS_pending_DO\\classes.txt')
