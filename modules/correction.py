"""系统校正模块 — 超前、滞后、滞后-超前校正器设计"""
import numpy as np
import control as ct
from modules.frequency import stability_margins


def _safe_crossover_freq(tf):
    """安全获取穿越频率，避免 NaN"""
    margins = stability_margins(tf)
    wg = margins.get('幅值穿越频率 ωg (rad/s)')
    wp = margins.get('相位穿越频率 ωp (rad/s)')

    if wg is not None and not np.isnan(wg) and wg > 0:
        return wg
    if wp is not None and not np.isnan(wp) and wp > 0:
        return wp
    poles = ct.poles(tf)
    if len(poles) > 0:
        mags = np.abs(poles)
        return max(mags.mean(), 0.1)
    return 1.0


def design_lead(tf, desired_pm=45, extra_phase=5):
    """
    超前校正器设计 Gc(s) = Kc * (Ts+1)/(alpha*Ts+1), alpha < 1
    返回: (gc_tf, corrected_tf, design_info)
    """
    margins = stability_margins(tf)
    current_pm = margins.get('相位裕度 Pm (°)', 0)
    if current_pm is None:
        current_pm = 0

    wm = _safe_crossover_freq(tf)

    phi_m = max(desired_pm - current_pm + extra_phase, 5.0)
    phi_m_rad = phi_m * np.pi / 180

    sin_phi = np.sin(phi_m_rad)
    if abs(sin_phi) > 1e-10:
        alpha = (1 - sin_phi) / (1 + sin_phi)
        alpha = max(0.01, min(alpha, 0.99))
    else:
        alpha = 0.1

    denom = wm * np.sqrt(alpha)
    if denom < 1e-10:
        denom = 0.1
    T = 1.0 / denom
    if np.isnan(T) or np.isinf(T) or T <= 0:
        T = 1.0

    num = [T, 1]
    den = [alpha * T, 1]
    gc = ct.TransferFunction(num, den)
    corrected = ct.series(gc, tf)

    return gc, corrected, {
        '类型': '超前校正',
        'alpha': alpha,
        'T': T,
        'phim (deg)': phi_m,
        '原相位裕度 (deg)': current_pm,
    }


def design_lag(tf, desired_pm=45, extra_phase=5):
    """
    滞后校正器设计 Gc(s) = Kc * (Ts+1)/(beta*Ts+1), beta > 1
    返回: (gc_tf, corrected_tf, design_info)
    """
    margins = stability_margins(tf)
    current_pm = margins.get('相位裕度 Pm (°)', 0)
    if current_pm is None:
        current_pm = 0

    wm = _safe_crossover_freq(tf)

    beta = 10.0
    T = 10.0 / wm
    if np.isnan(T) or np.isinf(T) or T <= 0:
        T = 10.0

    num = [T, 1]
    den = [beta * T, 1]
    gc = ct.TransferFunction(num, den)
    corrected = ct.series(gc, tf)

    return gc, corrected, {
        '类型': '滞后校正',
        'beta': beta,
        'T': T,
        '原相位裕度 (deg)': current_pm,
    }


def design_lag_lead(tf, desired_pm=45):
    """
    滞后-超前校正器设计
    Gc(s) = (T1s+1)(T2s+1) / ((beta*T1s+1)(alpha*T2s+1))
    返回: (gc_tf, corrected_tf, design_info)
    """
    margins = stability_margins(tf)
    current_pm = margins.get('相位裕度 Pm (°)', 0)
    if current_pm is None:
        current_pm = 0

    wm = _safe_crossover_freq(tf)

    phi_m = max(desired_pm - current_pm + 5, 5.0)
    phi_m_rad = phi_m * np.pi / 180
    sin_phi = np.sin(phi_m_rad)
    if abs(sin_phi) > 1e-10:
        alpha = (1 - sin_phi) / (1 + sin_phi)
        alpha = max(0.01, min(alpha, 0.99))
    else:
        alpha = 0.1

    beta = 10.0

    T1 = 10.0 / wm
    if np.isnan(T1) or np.isinf(T1) or T1 <= 0:
        T1 = 10.0

    denom2 = wm * np.sqrt(alpha)
    if denom2 < 1e-10:
        denom2 = 0.1
    T2 = 1.0 / denom2
    if np.isnan(T2) or np.isinf(T2) or T2 <= 0:
        T2 = 1.0

    num = [T1 * T2, T1 + T2, 1]
    den = [beta * T1 * alpha * T2, beta * T1 + alpha * T2, 1]
    gc = ct.TransferFunction(num, den)
    corrected = ct.series(gc, tf)

    return gc, corrected, {
        '类型': '滞后-超前校正',
        'alpha': alpha,
        'beta': beta,
        'T1': T1,
        'T2': T2,
        '原相位裕度 (deg)': current_pm,
    }


def compare_margins(tf_before, tf_after):
    """对比校正前后的稳定裕度"""
    before = stability_margins(tf_before)
    after = stability_margins(tf_after)
    return before, after
