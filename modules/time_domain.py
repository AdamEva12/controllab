"""时域分析模块 — 响应仿真、动态指标、稳态误差"""
import numpy as np
import control as ct
from modules.plotting import set_style


def step_response(tf, T=None):
    """阶跃响应，返回 (t, y, info_dict)"""
    if T is None:
        info = ct.step_info(tf)
        T = np.linspace(0, info.get('SettlingTime', 5) * 2, 1000)
    t, y = ct.step_response(tf, T)
    metrics = compute_metrics(t, y, 'step')
    return t, y, metrics


def ramp_response(tf, T=None):
    """斜坡响应，返回 (t, y, y_ideal)"""
    if T is None:
        info = ct.step_info(tf)
        T = np.linspace(0, info.get('SettlingTime', 5) * 2, 1000)

    t_ramp = T
    u_ramp = t_ramp

    t_sim, y_sim = ct.forced_response(tf, t_ramp, u_ramp)
    return t_sim, y_sim, u_ramp


def accel_response(tf, T=None):
    """加速度响应（抛物线输入 r(t) = t²/2），返回 (t, y, y_ideal)"""
    if T is None:
        info = ct.step_info(tf)
        T = np.linspace(0, info.get('SettlingTime', 5) * 2, 1000)

    u_accel = 0.5 * T ** 2
    t_sim, y_sim = ct.forced_response(tf, T, u_accel)
    return t_sim, y_sim, u_accel


def compute_metrics(t, y, input_type='step'):
    """
    计算动态性能指标。
    仅对阶跃响应计算超调量、峰值时间、上升时间、调节时间。
    """
    metrics = {}
    y_final = y[-1] if len(y) > 0 else 0

    if input_type != 'step' or abs(y_final) < 1e-10:
        metrics['稳态值'] = float(y_final)
        return metrics

    metrics['稳态值'] = float(y_final)
    y_norm = y / y_final

    i_peak = np.argmax(y)
    metrics['峰值时间 tp'] = float(t[i_peak])
    y_peak = y[i_peak]
    overshoot = (y_peak - y_final) / abs(y_final) * 100
    metrics['超调量 σ%'] = float(overshoot) if overshoot > 0.5 else 0.0

    try:
        i_10 = np.where(y_norm >= 0.1)[0][0]
        i_90 = np.where(y_norm >= 0.9)[0][0]
        metrics['上升时间 tr'] = float(t[i_90] - t[i_10])
    except IndexError:
        metrics['上升时间 tr'] = None

    tol_band = 0.02
    settled = np.where(np.abs(y_norm - 1) > tol_band)[0]
    if len(settled) > 0:
        i_settle = settled[-1] + 1
        if i_settle < len(t):
            metrics['调节时间 ts (±2%)'] = float(t[i_settle])
        else:
            metrics['调节时间 ts (±2%)'] = float(t[-1])
    else:
        metrics['调节时间 ts (±2%)'] = 0.0

    tol_band_5 = 0.05
    settled_5 = np.where(np.abs(y_norm - 1) > tol_band_5)[0]
    if len(settled_5) > 0:
        i_settle_5 = settled_5[-1] + 1
        if i_settle_5 < len(t):
            metrics['调节时间 ts (±5%)'] = float(t[i_settle_5])
        else:
            metrics['调节时间 ts (±5%)'] = float(t[-1])
    else:
        metrics['调节时间 ts (±5%)'] = 0.0

    try:
        i_50 = np.where(y_norm >= 0.5)[0][0]
        metrics['延迟时间 td'] = float(t[i_50])
    except IndexError:
        metrics['延迟时间 td'] = None

    return metrics


def steady_state_error(tf, input_type='step'):
    """
    计算稳态误差。
    对于单位反馈系统: E(s) = R(s) / (1 + G(s))
    ess = lim_{s→0} s·E(s)
    """
    try:
        if input_type == 'step':
            kp = ct.dcgain(tf)
            if kp == float('inf') or abs(kp) > 1e10:
                ess = 0.0
            else:
                ess = 1.0 / (1.0 + float(kp))
        elif input_type == 'ramp':
            s = ct.TransferFunction([1, 0], [1])
            g_s = ct.series(s, tf)
            kv = ct.dcgain(g_s)
            if abs(kv) < 1e-12 or kv == float('inf'):
                ess = float('inf')
            else:
                ess = 1.0 / float(kv)
        elif input_type == 'accel':
            s2 = ct.TransferFunction([1, 0, 0], [1])
            g_s2 = ct.series(s2, tf)
            ka = ct.dcgain(g_s2)
            if abs(ka) < 1e-12 or ka == float('inf'):
                ess = float('inf')
            else:
                ess = 1.0 / float(ka)
        else:
            ess = None
        return ess
    except Exception:
        return None


def plot_response(ax, t, y, label, color, input_signal=None):
    """绘制响应曲线"""
    ax.plot(t, y, color=color, linewidth=1.5, label=label)
    if input_signal is not None:
        ax.plot(t, input_signal, 'k--', linewidth=0.8, alpha=0.5, label='输入信号')
    set_style(ax, xlabel='时间 t (s)', ylabel='幅值')


def plot_pole_zero(ax, tf):
    """绘制零极点分布图"""
    poles = ct.poles(tf)
    zeros = ct.zeros(tf)

    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)

    if len(poles) > 0:
        ax.plot(poles.real, poles.imag, 'x', color='red', markersize=8,
                markeredgewidth=2, label=f'极点 ({len(poles)})')
    if len(zeros) > 0:
        ax.plot(zeros.real, zeros.imag, 'o', color='blue', markersize=8,
                markerfacecolor='none', markeredgewidth=2, label=f'零点 ({len(zeros)})')

    all_pts = np.concatenate([poles, zeros]) if len(zeros) > 0 else poles
    if len(all_pts) > 0:
        max_val = max(abs(all_pts.real).max(), abs(all_pts.imag).max(), 1) * 1.5
    else:
        max_val = 2
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_aspect('equal')
    set_style(ax, title='零极点分布图', xlabel='实轴 Re', ylabel='虚轴 Im')
    ax.legend(fontsize=9)
