"""根轨迹分析模块"""
import numpy as np
import control as ct
import matplotlib.pyplot as plt
from modules.plotting import set_style


def compute_root_locus(tf, kvect=None):
    """计算根轨迹数据"""
    if kvect is None:
        kvect = np.logspace(-2, 3, 2000)
    try:
        roots, gains = ct.root_locus(tf, gains=kvect, plot=False)
    except TypeError:
        roots, gains = ct.root_locus(tf, kvect=kvect, plot=False)
    return roots, gains


def plot_root_locus(ax, tf, kvect=None):
    """绘制根轨迹图"""
    roots, gains = compute_root_locus(tf, kvect)

    for i in range(roots.shape[1]):
        ax.plot(roots[:, i].real, roots[:, i].imag, 'b-', linewidth=0.8, alpha=0.7)

    poles = ct.poles(tf)
    ax.plot(poles.real, poles.imag, 'rx', markersize=8, markeredgewidth=2, label=f'开环极点 (K→0)')

    zeros = ct.zeros(tf)
    ax.plot(zeros.real, zeros.imag, 'bo', markersize=8, markerfacecolor='none',
            markeredgewidth=2, label=f'开环零点 (K→∞)')

    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)

    all_pts = np.concatenate([roots.flatten().real, roots.flatten().imag])
    if len(poles) > 0:
        all_pts = np.concatenate([all_pts, poles.real, poles.imag])
    max_v = max(abs(all_pts).max() * 1.2, 2)
    ax.set_xlim(-max_v, max_v)
    ax.set_ylim(-max_v, max_v)
    ax.set_aspect('equal')
    set_style(ax, title='根轨迹图 (Root Locus)', xlabel='实轴 Re', ylabel='虚轴 Im')
    ax.legend(fontsize=8, loc='best')

    return roots, gains


def find_gain_at_point(tf, target_point):
    """查找根轨迹上某点对应的增益"""
    try:
        r, k = ct.root_locus(tf, plot=False)
        dists = np.abs(r - target_point)
        idx = np.unravel_index(np.argmin(dists), r.shape)
        return k[idx[0]], r[idx[0], idx[1]]
    except Exception:
        return None, None
