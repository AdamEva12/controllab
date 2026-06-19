"""统一绘图封装 — 中文字体、样式、导出"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os


def _find_chinese_font():
    """查找系统中可用的中文字体"""
    candidates = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
                  'Noto Sans CJK SC', 'Source Han Sans SC', 'AR PL UMing CN']
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    for f in fm.fontManager.ttflist:
        if 'CJK' in f.name or 'Hei' in f.name or 'Song' in f.name:
            return f.name
    return None


_font_name = _find_chinese_font()
if _font_name:
    plt.rcParams['font.sans-serif'] = [_font_name, 'DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120


def set_style(ax, title='', xlabel='', ylabel='', grid=True):
    """统一设置坐标轴样式"""
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=9)


def fig_to_bytes(fig):
    """将 matplotlib figure 转为 PNG 字节流"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


FONT_OK = _font_name is not None
