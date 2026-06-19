"""自动控制原理仿真系统 - FastAPI 后端"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import control as ct
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="ControlLab - 自动控制原理仿真平台")

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


def _safe_float(v):
    """将 inf/nan 转为 None，确保 JSON 可序列化"""
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


def _get_tf(data: dict):
    """从请求数据构建传递函数"""
    if data.get("input_type") == "zpk":
        zeros = data.get("zeros", "")
        poles = data.get("poles", "")
        gain = float(data.get("gain", 1.0))
        if not poles.strip():
            return None, "至少需要输入一个极点"
        z = zeros.strip() or ""
        from modules.modeling import build_tf_zpk
        return build_tf_zpk(z, poles, gain)
    else:
        num = data.get("num", "1")
        den = data.get("den", "1 0.5 1")
        from modules.modeling import build_tf
        return build_tf(num, den)


@app.post("/api/tf/build")
async def api_build_tf(data: dict):
    from modules.modeling import tf_info
    tf, err = _get_tf(data)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    info = tf_info(tf)
    info_clean = {}
    for k, v in info.items():
        if isinstance(v, np.ndarray):
            info_clean[k] = [{"real": float(p.real), "imag": float(p.imag)} for p in v]
        elif isinstance(v, (np.floating, np.integer)):
            info_clean[k] = float(v)
        elif isinstance(v, complex):
            info_clean[k] = {"real": v.real, "imag": v.imag}
        else:
            info_clean[k] = v
    return info_clean


@app.post("/api/time/step")
async def api_step(data: dict):
    from modules.time_domain import step_response, steady_state_error
    tf, err = _get_tf(data)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    t_end = float(data.get("t_end", 20))
    T = np.linspace(0, t_end, 2000)
    t, y, metrics = step_response(tf, T)
    metrics_clean = {}
    for k, v in metrics.items():
        if v is None:
            metrics_clean[k] = None
        elif isinstance(v, (np.floating, np.integer, float, int)):
            metrics_clean[k] = _safe_float(v)
        else:
            metrics_clean[k] = v
    ess = steady_state_error(tf, 'step')
    return {
        "t": t.tolist(),
        "y": y.tolist(),
        "metrics": metrics_clean,
        "ess": _safe_float(ess)
    }


@app.post("/api/time/ramp")
async def api_ramp(data: dict):
    from modules.time_domain import ramp_response, steady_state_error
    tf, err = _get_tf(data)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    t_end = float(data.get("t_end", 20))
    T = np.linspace(0, t_end, 2000)
    t, y, u = ramp_response(tf, T)
    ess = steady_state_error(tf, 'ramp')
    return {
        "t": t.tolist(),
        "y": y.tolist(),
        "u": u.tolist(),
        "ess": _safe_float(ess)
    }


@app.post("/api/time/accel")
async def api_accel(data: dict):
    from modules.time_domain import accel_response, steady_state_error
    tf, err = _get_tf(data)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    t_end = float(data.get("t_end", 20))
    T = np.linspace(0, t_end, 2000)
    t, y, u = accel_response(tf, T)
    ess = steady_state_error(tf, 'accel')
    return {
        "t": t.tolist(),
        "y": y.tolist(),
        "u": u.tolist(),
        "ess": _safe_float(ess)
    }


@app.post("/api/polezero")
async def api_polezero(data: dict):
    tf, err = _get_tf(data)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    poles = ct.poles(tf)
    zeros = ct.zeros(tf)
    return {
        "poles": [{"real": float(p.real), "imag": float(p.imag)} for p in poles],
        "zeros": [{"real": float(z.real), "imag": float(z.imag)} for z in zeros],
    }


@app.post("/api/rlocus")
async def api_rlocus(data: dict):
    from modules.root_locus import compute_root_locus
    tf, err = _get_tf(data)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    k_min = float(data.get("k_min", 0.01))
    k_max = float(data.get("k_max", 100))
    kvect = np.logspace(np.log10(max(k_min, 1e-6)), np.log10(max(k_max, k_min + 0.1)), 1000)
    roots, gains = compute_root_locus(tf, kvect)
    traces = []
    for i in range(roots.shape[1]):
        traces.append({
            "real": roots[:, i].real.tolist(),
            "imag": roots[:, i].imag.tolist(),
            "k": [float(g) for g in gains]
        })
    poles = ct.poles(tf)
    zeros = ct.zeros(tf)
    return {
        "traces": traces,
        "poles_open": [{"real": float(p.real), "imag": float(p.imag)} for p in poles],
        "zeros_open": [{"real": float(z.real), "imag": float(z.imag)} for z in zeros],
        "k_min": float(k_min),
        "k_max": float(k_max),
    }


@app.post("/api/bode")
async def api_bode(data: dict):
    from modules.frequency import bode_data
    tf, err = _get_tf(data)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    w_min = float(data.get("w_min", 0.01))
    w_max = float(data.get("w_max", 100))
    omega = np.logspace(np.log10(max(w_min, 0.001)), np.log10(max(w_max, w_min + 0.1)), 1000)
    w, mag, phase = bode_data(tf, omega)
    mag_db = (20 * np.log10(mag)).tolist()
    phase_deg = (phase * 180 / np.pi).tolist()
    return {
        "omega": w.tolist(),
        "mag_db": mag_db,
        "phase_deg": phase_deg,
    }


@app.post("/api/nyquist")
async def api_nyquist(data: dict):
    from modules.frequency import nyquist_data
    tf, err = _get_tf(data)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    w_min = float(data.get("w_min", 0.01))
    w_max = float(data.get("w_max", 100))
    omega = np.logspace(np.log10(max(w_min, 0.001)), np.log10(max(w_max, w_min + 0.1)), 500)
    w, real, imag = nyquist_data(tf, omega)
    return {
        "real": real.tolist(),
        "imag": imag.tolist(),
        "omega": w.tolist(),
    }


@app.post("/api/margins")
async def api_margins(data: dict):
    from modules.frequency import stability_margins
    tf, err = _get_tf(data)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    margins = stability_margins(tf)
    clean = {}
    for k, v in margins.items():
        clean[k] = _safe_float(v)
    return clean


@app.post("/api/correction/design")
async def api_correction_design(data: dict):
    from modules.correction import design_lead, design_lag, design_lag_lead, compare_margins
    from modules.frequency import bode_data
    tf, err = _get_tf(data)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    corr_type = data.get("type", "lead")
    desired_pm = float(data.get("desired_pm", 50))
    extra = float(data.get("extra_phase", 5))

    if corr_type == "lead":
        gc, corrected, info = design_lead(tf, desired_pm, extra)
    elif corr_type == "lag":
        gc, corrected, info = design_lag(tf, desired_pm, extra)
    elif corr_type == "lag-lead":
        gc, corrected, info = design_lag_lead(tf, desired_pm)
    else:
        return JSONResponse({"error": "Unknown correction type"}, status_code=400)

    info_clean = {}
    for k, v in info.items():
        if isinstance(v, (np.floating, np.integer, float, int)):
            info_clean[k] = _safe_float(v)
        elif isinstance(v, np.ndarray):
            info_clean[k] = [_safe_float(x) for x in v.tolist()]
        else:
            info_clean[k] = v

    gc_num = gc.num[0][0].tolist()
    gc_den = gc.den[0][0].tolist()

    try:
        t_b, y_b = ct.step_response(ct.feedback(tf, 1))
        t_a, y_a = ct.step_response(ct.feedback(corrected, 1))
        step_comp = {
            "t_before": t_b.tolist(),
            "y_before": y_b.tolist(),
            "t_after": t_a.tolist(),
            "y_after": y_a.tolist(),
        }
    except Exception:
        step_comp = None

    w_b, mag_b, ph_b = bode_data(tf)
    w_a, mag_a, ph_a = bode_data(corrected)
    bode_comp = {
        "omega": w_b.tolist(),
        "mag_before": (20 * np.log10(mag_b)).tolist(),
        "phase_before": (ph_b * 180 / np.pi).tolist(),
        "mag_after": (20 * np.log10(mag_a)).tolist(),
        "phase_after": (ph_a * 180 / np.pi).tolist(),
    }

    before_m, after_m = compare_margins(tf, corrected)
    margins_comp = {
        "before": {k: _safe_float(v) for k, v in before_m.items()},
        "after": {k: _safe_float(v) for k, v in after_m.items()},
    }

    return {
        "info": info_clean,
        "gc_num": gc_num,
        "gc_den": gc_den,
        "step_comparison": step_comp,
        "bode_comparison": bode_comp,
        "margins_comparison": margins_comp,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8501))
    uvicorn.run(app, host="0.0.0.0", port=port)
