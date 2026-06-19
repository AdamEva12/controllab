"""频域分析模块 — Bode 图、Nyquist 图、稳定裕度"""
import numpy as np
import control as ct
import matplotlib.pyplot as plt
from modules.plotting import set_style


def bode_data(tf, omega=None):
    """计算 Bode 图数据"""
    if omega is None:
        omega = np.logspace(-2, 3, 1000)
    mag, phase, omega = ct.frequency_response(tf, omega)
    return omega, mag, phase


def nyquist_data(tf, omega=None):
    """计算 Nyquist 图数据"""
    if omega is None:
        omega = np.logspace(-2, 3, 1000)
    mag, phase, omega = ct.frequency_response(tf, omega)
    real = mag * np.cos(phase)
    imag = mag * np.sin(phase)
    return omega, real, imag


def stability_margins(tf):
    """计算稳定裕度"""
    try:
        gm, pm, wg, wp = ct.margin(tf)
        return {
            '幅值裕度 Gm (dB)': float(20 * np.log10(gm)) if gm is not None and gm > 0 else None,
            '相位裕度 Pm (°)': float(pm) if pm is not None else None,
            '幅值穿越频率 ωg (rad/s)': float(wg) if wg is not None else None,
            '相位穿越频率 ωp (rad/s)': float(wp) if wp is not None else None,
        }
    except Exception:
        return {}


def plot_bode(ax_mag, ax_phase, tf, omega=None):
    """在给定的两个子图上绘制 Bode 图"""
    omega, mag, phase = bode_data(tf, omega)
    mag_db = 20 * np.log10(mag)
    phase_deg = phase * 180 / np.pi

    ax_mag.semilogx(omega, mag_db, 'b-', linewidth=1.5)
    ax_mag.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')
    set_style(ax_mag, title='Bode 图', ylabel='幅值 (dB)')
    ax_mag.grid(True, which='both', alpha=0.3)
    ax_mag.grid(True, which='minor', alpha=0.1)

    ax_phase.semilogx(omega, phase_deg, 'b-', linewidth=1.5)
    ax_phase.axhline(y=-180, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')
    ax_phase.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')
    set_style(ax_phase, xlabel='频率 ω (rad/s)', ylabel='相位 (°)')
    ax_phase.grid(True, which='both', alpha=0.3)
    ax_phase.grid(True, which='minor', alpha=0.1)

    return omega, mag_db, phase_deg


def plot_nyquist(ax, tf, omega=None):
    """绘制 Nyquist 图"""
    omega, real, imag = nyquist_data(tf, omega)

    ax.plot(real, imag, 'b-', linewidth=1.2)
    ax.plot(real, -imag, 'b-', linewidth=0.6, alpha=0.3)

    ax.plot(-1, 0, 'rx', markersize=10, markeredgewidth=2)
    ax.annotate('(-1, j0)', (-1, 0), textcoords="offset points", xytext=(8, 8), fontsize=8, color='red')

    if len(real) > 0:
        ax.plot(real[0], imag[0], 'go', markersize=6, label=f'ω→0 ({omega[0]:.2f})')
        ax.plot(real[-1], imag[-1], 'mo', markersize=6, label=f'ω→∞ ({omega[-1]:.2f})')

    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_aspect('equal')
    set_style(ax, title='Nyquist 图', xlabel='实部 Re', ylabel='虚部 Im')
    ax.legend(fontsize=8, loc='best')
