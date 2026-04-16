"""
TFT_YOLOv11 每日工作流脚本

用法：
  python merge_and_build.py              # 归档 pending + 重建 datasets
  python merge_and_build.py --dry-run    # 只看计划，不实际执行
  python merge_and_build.py --val 5      # 指定验证集比例（默认15，即85:15）
  python merge_and_build.py --status     # 查看当前数据集状态

流程：
  1. 自动归档 pending/ + pending_DO/ -> TFT_CHESS + TFT_CHESS_DO
  2. 从 TFT_CHESS_DO 读取全量标注，TFT_CHESS 找图，拆分重建 datasets/

目录说明：
  C:/TFT_CHESS/            <- 所有截图存档
  C:/TFT_CHESS_pending/    <- 新截图（待预标）
  C:/TFT_CHESS_pending_DO/ <- pending 的预标结果（待确认）
  C:/TFT_CHESS_DO/         <- 所有标注（历史+归档）
"""

import os
import sys
import shutil
import random
import argparse
from datetime import datetime

# 添加项目根目录到 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ========== 路径配置 ==========
PATHS = {
    'all_labels':   r'C:\TFT_CHESS_DO',
    'all_screens':  r'C:\TFT_CHESS',
    'pending':      r'C:\TFT_CHESS_pending',
    'pending_DO':   r'C:\TFT_CHESS_pending_DO',
    'ds_train_img': os.path.join(PROJECT_ROOT, 'datasets', 'images', 'train'),
    'ds_train_lbl': os.path.join(PROJECT_ROOT, 'datasets', 'labels', 'train'),
    'ds_val_img':   os.path.join(PROJECT_ROOT, 'datasets', 'images', 'val'),
    'ds_val_lbl':   os.path.join(PROJECT_ROOT, 'datasets', 'labels', 'val'),
}

CLASSES_FILE = r'C:\TFT_CHESS_DO\classes.txt'


# ========== 工具函数 ==========

def get_labels(dir_path):
    if not os.path.exists(dir_path):
        return []
    return sorted([
        f for f in os.listdir(dir_path)
        if f.endswith('.txt') and f not in ('classes.txt', 'classes.names')
    ])


def get_images(dir_path):
    if not os.path.exists(dir_path):
        return []
    exts = ('.png', '.jpg', '.jpeg')
    return sorted(f for f in os.listdir(dir_path)
                  if f.lower().endswith(exts))


def find_image(base, search_dirs):
    for ext in ('.png', '.jpg', '.jpeg'):
        for d in search_dirs:
            p = os.path.join(d, base + ext)
            if os.path.exists(p):
                return p
    return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ========== classes.txt 检测 ==========

def read_classes():
    """读取 classes.txt（自动检测编码），返回类别列表。"""
    from utils._encoding import read_lines
    if not os.path.exists(CLASSES_FILE):
        return []
    return read_lines(CLASSES_FILE)


def detect_max_class_id(label_dir):
    """扫描目录下所有 label，找出最大的 class_id。"""
    max_id = -1
    from utils._encoding import read_text
    for fname in get_labels(label_dir):
        path = os.path.join(label_dir, fname)
        try:
            content, _ = read_text(path)
            for line in content.splitlines():
                parts = line.strip().split()
                if parts:
                    try:
                        cid = int(parts[0])
                        if cid > max_id:
                            max_id = cid
                    except ValueError:
                        pass
        except Exception:
            pass
    return max_id


def check_classes_warning():
    """
    扫描 TFT_CHESS_DO，检查是否有 label 用了超出 classes.txt 范围的 class_id。
    如果有，打印警告并列出缺失的 id，终止合并流程。
    """
    classes = read_classes()
    if not classes:
        return []  # classes.txt 不存在，跳过检测

    known_nc = len(classes)
    max_id   = detect_max_class_id(PATHS['all_labels'])

    if max_id >= known_nc:
        missing = list(range(known_nc, max_id + 1))
        print('\n[WARN] 发现新类别！')
        print('  classes.txt 现有 %d 类（id 0-%d），' % (known_nc, known_nc - 1))
        print('  但标签中出现了 id %d' % max_id)
        print('  缺少 id: %s' % missing)
        print()
        print('  请打开 C:\\TFT_CHESS_DO\\classes.txt 补充类别名，')
        print('  再重新运行 merge_and_build.py')
        print()
        return missing
    return []


# ========== 核心逻辑 ==========

def archive_pending():
    """
    将 pending/ + pending_DO/ 归档到 TFT_CHESS + TFT_CHESS_DO，
    完成后清空两个 pending 目录。
    返回归档的标签数量。
    """
    pending_dir = PATHS['pending']
    pending_do  = PATHS['pending_DO']

    pending_imgs = get_images(pending_dir)
    pending_lbls = {f.rsplit('.', 1)[0]: f
                    for f in os.listdir(pending_do) if f.endswith('.txt')
                    and f not in ('classes.txt', 'classes.names')}

    if not pending_imgs and not pending_lbls:
        print('  [STEP 1] pending/ 和 pending_DO/ 都为空，跳过归档')
        return 0

    added_imgs, added_lbls = 0, 0
    skipped_imgs, skipped_lbls = [], []

    # 图：pending/ -> TFT_CHESS
    for img_file in pending_imgs:
        dst = os.path.join(PATHS['all_screens'], img_file)
        if not os.path.exists(dst):
            shutil.copy(os.path.join(pending_dir, img_file), dst)
            added_imgs += 1
        else:
            skipped_imgs.append(img_file)

    # 标签：pending_DO/ -> TFT_CHESS_DO（classes.txt 不动）
    for base, lbl_file in pending_lbls.items():
        dst = os.path.join(PATHS['all_labels'], lbl_file)
        if not os.path.exists(dst):
            shutil.copy(os.path.join(pending_do, lbl_file), dst)
            added_lbls += 1
        else:
            skipped_lbls.append(lbl_file)

    # 清空 pending/
    for f in os.listdir(pending_dir):
        os.remove(os.path.join(pending_dir, f))

    # 清空 pending_DO/（保留 classes.txt）
    for f in os.listdir(pending_do):
        if f not in ('classes.txt', 'classes.names'):
            os.remove(os.path.join(pending_do, f))

    print('  [STEP 1] 归档完成：+%d 图 -> TFT_CHESS，+%d 标签 -> TFT_CHESS_DO' % (added_imgs, added_lbls))
    if skipped_imgs:
        print('           跳过（图已存在）: %d 张' % len(skipped_imgs))
    if skipped_lbls:
        print('           跳过（标签已存在）: %d 个' % len(skipped_lbls))
    return added_lbls


def merge_and_build(val_ratio=15):
    """
    Step 1: 自动归档 pending/ + pending_DO/ -> TFT_CHESS + TFT_CHESS_DO
    Step 2: 读取 TFT_CHESS_DO 全量标注，从 TFT_CHESS 找图，拆分重建 datasets/
    重建前自动检测是否有新类别缺失。
    """
    # Step 1: 归档 pending 数据
    archive_pending()

    # Step 2: 重建 datasets
    label_dir = PATHS['all_labels']
    if not os.path.exists(label_dir):
        print('[!] C:\\TFT_CHESS_DO 不存在')
        return

    print('  [STEP 2] 检测新类别...')

    # Step 2: 检测新类别
    missing_ids = check_classes_warning()
    if missing_ids:
        print('[!] 检测到 %d 个新类别尚未录入 classes.txt，终止本次合并'
              % len(missing_ids))
        return

    labels      = get_labels(label_dir)
    search_dirs = [PATHS['all_screens'], PATHS['all_labels']]

    print('  [STEP 3] 重建 datasets（验证集 %d%%）...' % int(val_ratio))

    random.seed(20241115)
    random.shuffle(labels)
    random.seed()

    n_val        = max(1, int(len(labels) * val_ratio / 100))
    val_files    = set(labels[:n_val])
    train_files  = set(labels[n_val:])

    for d in [PATHS['ds_train_img'], PATHS['ds_train_lbl'],
              PATHS['ds_val_img'],   PATHS['ds_val_lbl']]:
        if os.path.exists(d):
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
        else:
            ensure_dir(d)

    ok_train, ok_val = 0, 0
    missing_imgs = []

    def copy_one(fname, dest_img_dir, dest_lbl_dir):
        nonlocal ok_train, ok_val, missing_imgs
        base = fname.replace('.txt', '')
        img  = find_image(base, search_dirs)
        lbl  = os.path.join(label_dir, fname)
        if img:
            shutil.copy(img, os.path.join(dest_img_dir, os.path.basename(img)))
            shutil.copy(lbl, os.path.join(dest_lbl_dir, fname))
            if dest_img_dir == PATHS['ds_train_img']:
                ok_train += 1
            else:
                ok_val += 1
        else:
            missing_imgs.append(base)

    for f in train_files: copy_one(f, PATHS['ds_train_img'], PATHS['ds_train_lbl'])
    for f in val_files:   copy_one(f, PATHS['ds_val_img'],   PATHS['ds_val_lbl'])

    print('\n[OK] datasets 重建完成！')
    print('  C:\\TFT_CHESS_DO 共 %d 个标注 -> 训练集 %d | 验证集 %d'
          % (len(labels), ok_train, ok_val))
    if missing_imgs:
        print('  [WARN] %d 个标注找不到图片: %s'
              % (len(missing_imgs), missing_imgs[:3]))
    else:
        print('  所有标注均找到对应图片！')


def status():
    pending_imgs  = get_images(PATHS['pending'])
    pending_lbls  = get_labels(PATHS['pending_DO'])
    all_lbls     = get_labels(PATHS['all_labels'])
    train_imgs   = get_images(PATHS['ds_train_img'])
    val_imgs     = get_images(PATHS['ds_val_img'])
    classes      = read_classes()
    max_id       = detect_max_class_id(PATHS['all_labels'])

    print('\n========== 当前数据集状态 ==========')
    print('  C:\\TFT_CHESS            (截图存档)')
    print('  C:\\TFT_CHESS_pending    %d 图  <- 新截图（待预标）' % len(pending_imgs))
    print('  C:\\TFT_CHESS_pending_DO %d 标签 <- 待归档预标结果' % len(pending_lbls))
    print('  C:\\TFT_CHESS_DO         %d 标签  <- 所有标注' % len(all_lbls))
    print('  classes.txt              %d 类  <- 类别列表' % len(classes))
    print('  datasets/train           %d 图  <- 当前训练集' % len(train_imgs))
    print('  datasets/val             %d 图  <- 当前验证集' % len(val_imgs))
    if max_id >= len(classes) and classes:
        print('  [WARN] 标签中最大 id=%d，classes.txt 只有 %d 类（id 0-%d）'
              % (max_id, len(classes), len(classes) - 1))
    print('=' * 36)


def main():
    parser = argparse.ArgumentParser(description='TFT YOLO 每日工作流')
    parser.add_argument('--dry-run',  action='store_true', help='只打印计划，不实际执行')
    parser.add_argument('--val',      type=float, default=15, help='验证集比例%%（默认15，即85:15）')
    parser.add_argument('--status',   action='store_true', help='查看状态')
    args = parser.parse_args()

    print('=' * 50)
    print('TFT_YOLOv11 每日工作流')
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('=' * 50)

    if args.status:
        status()
        return

    if args.dry_run:
        pending_imgs = get_images(PATHS['pending'])
        pending_lbls = get_labels(PATHS['pending_DO'])
        print('\n[DRY RUN] merge_and_build 会执行：')
        print('  [STEP 1] 归档 pending: %d 图 + %d 标签' % (len(pending_imgs), len(pending_lbls)))
        n = len(get_labels(PATHS['all_labels'])) + len(pending_lbls)
        n_val = max(1, int(n * args.val / 100))
        print('  [STEP 2] 重建 datasets: 共 %d 标注 -> 训练集 %d | 验证集 %d' % (n, n - n_val, n_val))
        missing_ids = check_classes_warning()
        if missing_ids:
            print('  [WARN] 缺少类别 id: %s' % missing_ids)
        print('\n[dry-run，不做任何更改]')
        return

    merge_and_build(val_ratio=args.val)

    print('\n' + '=' * 50)
    print('下一步：')
    print('  1. cd scripts && python train.py      # 训练新模型')
    print('  2. python predict.py                  # 预标 pending 里的新图')
    print('=' * 50)


if __name__ == '__main__':
    main()
