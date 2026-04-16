"""
统一编码检测工具
读取文本文件时自动探测编码，优先 UTF-8，失败则尝试 GBK。
"""

def read_text(path, fallback_encoding='gbk'):
    """读取文本文件，优先 UTF-8，失败则 fallback 到 GBK。返回 (content, encoding)。"""
    for enc in ('utf-8', fallback_encoding, 'gb18030'):
        try:
            with open(path, 'r', encoding=enc) as f:
                content = f.read()
            return content, enc
        except (UnicodeDecodeError, LookupError):
            continue
    # 最后兜底：errors='replace'
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read(), 'utf-8-replace'


def write_text(path, content, encoding='utf-8'):
    """写入文本文件，统一 UTF-8（无 BOM）。"""
    with open(path, 'w', encoding=encoding) as f:
        f.write(content)


def read_lines(path, fallback_encoding='gbk'):
    """读取文本文件，返回非空、非注释行列表。"""
    content, enc = read_text(path, fallback_encoding)
    return [l.strip() for l in content.splitlines()
            if l.strip() and not l.strip().startswith('#')]


def detect_file_encoding(path):
    """检测文件编码，返回检测到的编码名。"""
    _, enc = read_text(path)
    return enc
