"""
TFT 棋子标注工具（批量版）

用法：
  python predict.py                          # 批量推理（不导出，让用户先核对）
  python predict.py --labels-only            # 导出 labeled/ 中已有的 .txt 标签
  python predict.py --preview                # 逐张预览模式

手动修正工作流（推荐）：
  1. python predict.py                        # 模型推理 → labeled/
  2. 打开 labeled/，用 labelimg 逐张核对/修改
  3. python predict.py --labels-only          # 导出 labelimg 的标注 → C:/TFT_CHESS_DO
  4. python merge_and_build.py                # 重建 datasets
  5. python train.py                          # 训练
"""

from ultralytics import YOLO
import os, cv2, numpy as np, shutil, sys

# ================= 配置区域 =================
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

model_path     = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'TFT_Result', 'my_tft_model', 'weights', 'best.pt')
test_folder    = r'C:\TFT_CHESS_pending'
labeled_folder = r'C:\TFT_CHESS_pending_DO'
label_out_dir  = r'C:\TFT_CHESS_DO'
IMG_EXTS       = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.35
DIST_THRESHOLD = 50   # 合并过近框（像素）
CLASSES_FILE   = r'C:\TFT_CHESS_DO\classes.txt'
# =============================================


# ================= 工具函数 =================

def merge_close_boxes(boxes, confs, classes, dist_thresh=50):
    """合并中心点距离 < dist_thresh 的框，只保留置信度最高的"""
    if len(boxes) <= 1:
        return boxes, confs, classes
    boxes_arr = np.array(boxes)
    if boxes_arr.ndim == 0:
        boxes_arr = np.atleast_2d(boxes_arr)
    centers = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in boxes_arr])
    used = set()
    keep = []
    for i in range(len(boxes_arr)):
        if i in used:
            continue
        for j in range(i+1, len(boxes_arr)):
            if j in used:
                continue
            if np.linalg.norm(centers[i] - centers[j]) < dist_thresh:
                used.add(j)
        keep.append(i)
        used.add(i)
    return [boxes[i] for i in keep], [confs[i] for i in keep], [classes[i] for i in keep]


def draw_boxes(img, boxes, confs, classes, names, scale=2):
    """在原图上画框（绿色）"""
    for b, conf, cls in zip(boxes, confs, classes):
        x1, y1, x2, y2 = map(int, b)
        label = f"{names[int(cls)]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), scale)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, scale)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), scale)
    return img


def sync_classes_to_labeled():
    """
    恢复 classes.txt 到 labeled/。
    把原始完整类列表（from C:/TFT_CHESS_DO/classes.txt）同步进 labeled/，
    覆盖 labelimg 截断后的脏 classes.txt，防止 IndexError。
    """
    src = r'C:\TFT_CHESS_DO\classes.txt'
    dst = os.path.join(labeled_folder, 'classes.txt')
    if not os.path.exists(src):
        return

    # 自动检测编码读取（兼容旧 GBK 文件）
    from utils._encoding import read_lines
    orig_classes = read_lines(src)

    # 写入 labeled/classes.txt（UTF-8，与 labelimg 读取编码一致）
    with open(dst, 'w', encoding='utf-8') as f:
        for c in orig_classes:
            f.write(c + '\n')


# ================= 推理入口 =================

def predict_batch():
    """
    批量推理模式：推理 test_folder 的图片 → labeled_folder，直接生成 .txt 标注
    """
    os.makedirs(labeled_folder, exist_ok=True)
    sync_classes_to_labeled()  # 为 labelimg 保持 classes.txt

    if not os.path.exists(model_path):
        print(f"[错误] 找不到模型: {model_path}")
        return
    model = YOLO(model_path)
    print(f"模型类别: {list(model.names.values())}")

    images = [f for f in os.listdir(test_folder)
              if os.path.splitext(f)[1].lower() in IMG_EXTS
              and not f.startswith('result_') and not f.startswith('labeled_')]

    if not images:
        print(f"[提示] {test_folder} 中没有待标注图片")
        return

    print(f"待标注: {len(images)} 张\n")

    total_boxes = 0
    total_labels = 0
    for i, img_file in enumerate(sorted(images), 1):
        img_path = os.path.join(test_folder, img_file)
        results  = model(img_path, conf=CONF_THRESHOLD, imgsz=1280,
                         rect=False, iou=IOU_THRESHOLD)
        r = results[0]

        boxes   = r.boxes.xyxy.cpu().numpy().tolist()
        confs   = r.boxes.conf.cpu().numpy().tolist()
        classes = r.boxes.cls.cpu().numpy().tolist()

        raw_count = len(boxes)
        if len(boxes) > 0:
            boxes, confs, classes = merge_close_boxes(boxes, confs, classes, DIST_THRESHOLD)

        # 生成 YOLO 格式 .txt 标注文件（文件名与原图一一对应）
        if len(boxes) > 0:
            img_h, img_w = r.orig_img.shape[:2]
            label_path = os.path.join(labeled_folder, img_file.rsplit('.', 1)[0] + '.txt')
            with open(label_path, 'w') as f:
                for b, cls in zip(boxes, classes):
                    x1, y1, x2, y2 = b
                    cx = ((x1 + x2) / 2) / img_w
                    cy = ((y1 + y2) / 2) / img_h
                    bw = (x2 - x1) / img_w
                    bh = (y2 - y1) / img_h
                    f.write(f'{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
            total_labels += 1

        tag = ''
        if classes:
            tag = '  [' + ', '.join(f'{r.names[int(c)]}' for c in sorted(set(int(c) for c in classes))) + ']'
        change = ' → 去重%02d个' % len(boxes) if raw_count != len(boxes) else ''
        print(f"[{i:02d}/{len(images)}] {img_file}  原始{raw_count:02d}{change}{tag}")

        total_boxes += len(boxes)

    print(f"\n标注完成！{len(images)} 张 → {labeled_folder}")
    print(f"共生成 {total_labels} 个 .txt 标注文件")
    print(f"共检测 {total_boxes} 个棋子")
    print(f"\n[下一步]")
    print(f"  1. python predict.py --labels-only   # 导出标注到 C:\\TFT_CHESS_DO")
    print()


def predict_preview():
    """逐张预览模式：推理 + 弹窗 + 保存（不导出）"""
    os.makedirs(labeled_folder, exist_ok=True)
    sync_classes_to_labeled()

    if not os.path.exists(model_path):
        print(f"[错误] 找不到模型: {model_path}")
        return
    model = YOLO(model_path)

    images = [f for f in os.listdir(test_folder)
              if os.path.splitext(f)[1].lower() in IMG_EXTS
              and not f.startswith('result_') and not f.startswith('labeled_')]

    if not images:
        print(f"[提示] {test_folder} 中没有图片")
        return

    print(f"共 {len(images)} 张，类别: {list(model.names.values())}")
    print(f"标注结果 → {labeled_folder}\n")

    for img_file in sorted(images):
        img_path = os.path.join(test_folder, img_file)
        print(f"▶ {img_file}")
        results  = model(img_path, conf=CONF_THRESHOLD, imgsz=1280,
                         rect=False, iou=IOU_THRESHOLD)
        r = results[0]

        boxes   = r.boxes.xyxy.cpu().numpy().tolist()
        confs   = r.boxes.conf.cpu().numpy().tolist()
        classes = r.boxes.cls.cpu().numpy().tolist()

        print(f"  原始: {len(boxes)} 个", end='')
        if len(boxes) > 0:
            boxes, confs, classes = merge_close_boxes(boxes, confs, classes, DIST_THRESHOLD)
            if len(r.boxes) != len(boxes):
                print(f" → 去重{len(boxes)} 个")
            else:
                print()
        if len(boxes) == 0:
            print("  [无检测]")
        else:
            for cid in sorted(set(int(c) for c in classes)):
                cnt = sum(1 for c in classes if int(c) == cid)
                print(f"    {r.names[cid]}: {cnt}")

        img_show = r.orig_img.copy()
        if len(boxes) > 0:
            img_show = draw_boxes(img_show, boxes, confs, classes, r.names)

        cv2.namedWindow('TFT 标注预览', cv2.WINDOW_NORMAL)
        cv2.imshow('TFT 标注预览', img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        save_path = os.path.join(labeled_folder, f'labeled_{img_file}')
        cv2.imwrite(save_path, img_show)
        print(f"  ✓ 保存 → {save_path}\n")


# ================= 导出标签 =================

def read_classes(classes_file=CLASSES_FILE):
    """读取类别列表（自动检测编码）。"""
    from utils._encoding import read_lines
    if not os.path.exists(classes_file):
        return []
    return read_lines(classes_file)


def write_classes(classes_list, classes_file=CLASSES_FILE):
    """写入类别列表（统一 UTF-8）。"""
    from utils._encoding import write_text
    content = '\n'.join(c.strip() for c in classes_list if c.strip())
    write_text(classes_file, content)


def auto_update_classes():
    """
    检测 C:/TFT_CHESS_DO 所有 label 中是否有超出 classes.txt 范围的 class_id。
    如果有，在命令行询问新类别名称，自动补全 classes.txt。
    返回是否做了更新。
    """
    classes = read_classes()
    if not classes:
        return False

    known_nc = len(classes)

    max_id = -1
    from utils._encoding import read_text
    for fname in os.listdir(label_out_dir):
        if not fname.endswith('.txt'):
            continue
        try:
            content, _ = read_text(os.path.join(label_out_dir, fname))
            for line in content.splitlines():
                parts = line.strip().split()
                if parts:
                    cid = int(parts[0])
                    if cid > max_id:
                        max_id = cid
        except Exception:
            pass

    if max_id < known_nc:
        return False

    missing_ids = list(range(known_nc, max_id + 1))
    print(f"\n[发现新类别] 标签中出现了 id {missing_ids}，"
          f"但 classes.txt 只有 {known_nc} 类（id 0-{known_nc-1}）")

    new_names = []
    for cid in missing_ids:
        name = input(f'发现新类别 id={cid}，请输入这个棋子的名称：').strip()
        if not name:
            name = f'新棋子{cid}'
        new_names.append(name)

    all_classes = classes + new_names
    write_classes(all_classes)
    print(f"\n[OK] classes.txt 已更新，共 {len(all_classes)} 类：{all_classes}")
    return True


def export_labels():
    """
    把 labeled_folder 里的 .txt 标注导出到 C:/TFT_CHESS_DO，
    同时把对应图片同步到 C:\\TFT_CHESS。
    不会覆盖已有标签，发现新类别时自动询问补全 classes.txt。
    """
    if not os.path.exists(labeled_folder):
        print("[提示] labeled/ 不存在，请先运行 python predict.py")
        return

    txt_files = {}
    for f in os.listdir(labeled_folder):
        ext = os.path.splitext(f)[1].lower()
        if ext == '.txt' and f != 'classes.txt':
            orig_base = os.path.splitext(f)[0]
            txt_files[orig_base] = f

    if not txt_files:
        print(f"[提示] {labeled_folder} 中没有找到 .txt 标签文件")
        return

    os.makedirs(label_out_dir, exist_ok=True)

    bak_dir = label_out_dir + '_bak'
    if os.path.exists(label_out_dir) and not os.path.exists(bak_dir):
        shutil.copytree(label_out_dir, bak_dir)
        print(f"[备份] 现有标签 → {bak_dir}")

    exported, skipped = 0, 0
    for orig_base, txt_file in sorted(txt_files.items()):
        src_path = os.path.join(labeled_folder, txt_file)

        img_src = None
        for ext in IMG_EXTS:
            p = os.path.join(labeled_folder, orig_base + ext)
            if os.path.exists(p):
                img_src = p
                break
        if img_src is None:
            for ext in IMG_EXTS:
                p2 = os.path.join(test_folder, orig_base + ext)
                if os.path.exists(p2):
                    img_src = p2
                    break

        dst_path = os.path.join(label_out_dir, orig_base + '.txt')
        if os.path.exists(dst_path):
            skipped += 1
        else:
            shutil.copy2(src_path, dst_path)

        if img_src is not None:
            img_dst = os.path.join(r'C:\TFT_CHESS', os.path.basename(img_src))
            if not os.path.exists(img_dst):
                shutil.copy2(img_src, img_dst)

        exported += 1

    print(f"标签导出完成！新增 {exported} 个，跳过（已有）{skipped} 个")
    print(f"图片已同步到 C:\\TFT_CHESS")

    updated = auto_update_classes()

    print("\n" + "=" * 50)
    if updated:
        print("classes.txt 已自动更新，直接执行：")
        print("  python merge_and_build.py")
        print("  python train.py")
    else:
        print("classes.txt 无需更新，直接执行：")
        print("  python merge_and_build.py")
        print("  python train.py")
    print("=" * 50)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == '--preview':
            predict_preview()
        elif arg == '--labels-only':
            export_labels()
        elif arg == '--restore-classes':
            sync_classes_to_labeled()
            print("[OK] classes.txt 已恢复")
        else:
            print("用法: python predict.py [--preview|--labels-only|--restore-classes]")
    else:
        predict_batch()
