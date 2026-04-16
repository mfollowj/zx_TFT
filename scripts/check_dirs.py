import os, sys
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

paths = {
    r'C:\TFT_CHESS':         'ORIG screenshots',
    r'C:\TFT_CHESS_DO':      'LABELS all manual',
    r'C:\TFT_CHESS_pending': 'PENDING new imgs',
    r'C:\TFT_CHESS_annotated': 'ANNOTATED checked labels',
    os.path.join(PROJECT_ROOT, 'datasets', 'images', 'train'):  'TRAIN imgs',
    os.path.join(PROJECT_ROOT, 'datasets', 'labels', 'train'):  'TRAIN lbls',
    os.path.join(PROJECT_ROOT, 'datasets', 'images', 'val'):    'VAL imgs',
    os.path.join(PROJECT_ROOT, 'datasets', 'labels', 'val'):    'VAL lbls',
}
for p, desc in paths.items():
    if os.path.exists(p):
        files = os.listdir(p)
        imgs = [f for f in files if f.lower().endswith(('.png','.jpg','.jpeg'))]
        txts = [f for f in files if f.endswith('.txt') and f not in ('classes.txt','classes.names')]
        print('[OK] %s | %s | %d imgs %d txts' % (p, desc, len(imgs), len(txts)))
    else:
        print('[NEW] %s | %s | will-create' % (p, desc))
