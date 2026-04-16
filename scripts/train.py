"""
TFT YOLO 训练脚本
启动时自动从 C:\TFT_CHESS_DO\classes.txt 读取类别列表，
同步更新 data.yaml（nc 和 names），无需手动维护 yaml。
"""

import os
import sys

# 添加项目根目录到 path，确保能找到 utils 模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import yaml
from ultralytics import YOLO

# ============================================================
# 启动时自动同步 classes.txt -> data.yaml
# ============================================================

CLASSES_FILE = r'C:\TFT_CHESS_DO\classes.txt'
DATA_YAML    = os.path.join(PROJECT_ROOT, 'data.yaml')
WEIGHTS_DIR  = os.path.join(PROJECT_ROOT, 'weights')


def sync_data_yaml():
    """读取 classes.txt，同步 data.yaml 的 nc 和 names"""
    if not os.path.exists(CLASSES_FILE):
        print('[WARN] classes.txt 不存在，跳过同步')
        return

    # classes.txt 可能是 GBK 或 UTF-8，尝试自动检测
    from utils._encoding import read_lines
    class_names = read_lines(CLASSES_FILE)

    if not class_names:
        print('[WARN] classes.txt 为空，跳过同步')
        return

    # 读取现有 yaml（保留 path / train / val 等其他字段）
    with open(DATA_YAML, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f) or {}

    old_names = yaml_data.get('names', [])
    old_nc    = yaml_data.get('nc', 0)

    # 检测变化
    if old_names == class_names and old_nc == len(class_names):
        print('[INFO] classes.txt 与 data.yaml 一致，无需更新')
        return

    yaml_data['nc']    = len(class_names)
    yaml_data['names'] = class_names

    with open(DATA_YAML, 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_data, f, allow_unicode=True, sort_keys=False)

    print('[OK] data.yaml 已同步！')
    print('     nc: %d  names: %s' % (yaml_data['nc'], class_names))


# ============================================================
# 训练入口
# ============================================================

if __name__ == '__main__':
    # Step 1: 同步类别
    sync_data_yaml()

    # Step 2: 加载模型
    print('\n正在加载 YOLOv8s 模型...')
    model = YOLO(os.path.join(WEIGHTS_DIR, 'yolov8s.pt'))

    # Step 3: 训练
    print('开始训练任务...')
    model.train(
        data=DATA_YAML,
        epochs=150,
        patience=25,
        imgsz=1280,
        batch=8,
        workers=2,
        lr0=0.001,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        project='TFT_Result',
        name='my_tft_model',
        plots=True
    )

    print('\n训练结束！请去 runs/detect/TFT_Result/my_tft_model/weights/ 找 best.pt')
