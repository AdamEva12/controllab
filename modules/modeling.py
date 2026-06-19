"""系统建模模块：传递函数构建、梅森增益公式"""
import control as ct
import numpy as np
import re
import time
from functools import lru_cache


def parse_polynomial(s: str):
    """
    解析实系数多项式字符串，返回系数列表（降幂排列）。
    支持格式: "1 2 1" 或 "1,2,1" 或 "1, 2, 1"
    仅用于多项式系数输入。
    """
    s = s.strip()
    if not s:
        return None
    parts = re.split(r'[\s,;]+', s)
    parts = [p for p in parts if p]
    if not parts:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def _parse_single_complex(token: str):
    """
    解析单个复数 token。
    支持格式: a+bj, a-bj, a+jb, a-jb, a+bi, a-bi, a+ib, a-ib, a (纯实数)
    """
    token = token.strip().replace(' ', '')
    if not token:
        return None

    # 纯实数
    try:
        return complex(float(token), 0)
    except ValueError:
        pass

    # 规范化虚数单位：i -> j
    token = token.replace('i', 'j')

    # Python complex() 直接处理标准格式 a+bj / a-bj
    try:
        return complex(token)
    except ValueError:
        pass

    # 处理 a+jb 格式 (j 在中间，后面跟数字)
    m = re.match(r'^([-+]?\d*\.?\d*(?:[eE][-+]?\d+)?)([-+])([jJ])(\d*\.?\d*)$', token)
    if m:
        real = float(m.group(1))
        sign = 1 if m.group(2) == '+' else -1
        imag = float(m.group(4)) if m.group(4) else 1.0
        return complex(real, sign * imag)

    # 处理 a-jb 格式 (上面应该已经匹配了)
    m = re.match(r'^([-+]?\d*\.?\d*(?:[eE][-+]?\d+)?)([-+])(\d*\.?\d*)([jJ])$', token)
    if m:
        real = float(m.group(1))
        sign = 1 if m.group(2) == '+' else -1
        imag = float(m.group(3))
        return complex(real, sign * imag)

    # 处理单独的虚数 jX / -jX
    m = re.match(r'^([-+]?)([jJ])(\d*\.?\d*)$', token)
    if m:
        sign = -1 if m.group(1) == '-' else 1
        imag = float(m.group(3)) if m.group(3) else 1.0
        return complex(0, sign * imag)

    return None


def parse_complex_list(s: str):
    """
    解析复数列表，支持逗号分隔。
    返回复数列表，解析失败返回 None。
    """
    s = s.strip()
    if not s:
        return []

    # 按逗号分隔（注意复数中的符号不要被分割）
    tokens = re.split(r'[,;，；]+', s)
    tokens = [t.strip() for t in tokens if t.strip()]
    if not tokens:
        return []

    result = []
    for token in tokens:
        c = _parse_single_complex(token)
        if c is None:
            return None
        result.append(c)
    return result


def build_tf(num_str: str, den_str: str):
    """
    根据分子分母多项式系数构建传递函数。
    返回 (tf_object, error_message)
    """
    num = parse_polynomial(num_str)
    den = parse_polynomial(den_str)

    if num is None or len(num) == 0:
        return None, "分子系数无效，请用空格或逗号分隔"
    if den is None or len(den) == 0:
        return None, "分母系数无效，请用空格或逗号分隔"
    if abs(den[0]) < 1e-12:
        return None, "分母首项系数不能为零"

    try:
        tf = ct.TransferFunction(num, den)
        return tf, None
    except Exception as e:
        return None, f"传递函数构建失败: {str(e)}"


def build_tf_zpk(zeros_str: str, poles_str: str, gain: float):
    """
    根据零极点（支持复数）构建传递函数。
    返回 (tf_object, error_message)
    """
    zeros = parse_complex_list(zeros_str) if zeros_str.strip() else []
    if zeros is None:
        return None, "零点格式无效，支持: -1, -2+3j, 0.5-0.8j"

    poles = parse_complex_list(poles_str)
    if poles is None:
        return None, "极点格式无效，支持: -0.25+0.97j, -0.25-0.97j"
    if len(poles) == 0:
        return None, "至少需要输入一个极点"

    try:
        tf = ct.zpk(zeros, poles, gain)
        return tf, None
    except Exception as e:
        return None, f"传递函数构建失败: {str(e)}"


def tf_to_latex(tf):
    """将传递函数转为 LaTeX 字符串"""
    num = tf.num[0][0]
    den = tf.den[0][0]

    def poly_str(coeffs, var='s'):
        terms = []
        n = len(coeffs) - 1
        for i, c in enumerate(coeffs):
            if abs(c) < 1e-12:
                continue
            power = n - i
            c_real = c.real if isinstance(c, complex) else c
            c_str = f'{c_real:.4g}'
            if power == 0:
                terms.append(c_str)
            elif power == 1:
                term = c_str if abs(c_real - 1) > 1e-12 and abs(c_real + 1) > 1e-12 else ('1' if c_real > 0 else '-1')
                terms.append(f'{term}s')
            else:
                term = c_str if abs(c_real - 1) > 1e-12 and abs(c_real + 1) > 1e-12 else ('1' if c_real > 0 else '-1')
                terms.append(f'{term}s^{power}')
        if not terms:
            return '0'
        s = terms[0]
        for t in terms[1:]:
            if t.startswith('-'):
                s += ' ' + t
            else:
                s += ' + ' + t
        return s

    return f'$$G(s) = \\frac{{{poly_str(num)}}}{{{poly_str(den)}}}$$'


def tf_info(tf):
    """获取传递函数的详细信息"""
    poles = ct.poles(tf)
    zeros = ct.zeros(tf)
    dc_gain = ct.dcgain(tf)

    info = {
        '分子': str(tf.num[0][0]),
        '分母': str(tf.den[0][0]),
        '极点': poles,
        '零点': zeros,
        '直流增益': float(dc_gain) if np.isscalar(dc_gain) else dc_gain,
        '阶数': len(tf.den[0][0]) - 1,
    }

    if all(p.real < 0 for p in poles):
        info['稳定性'] = '稳定 (所有极点在左半平面)'
    elif any(p.real > 0 for p in poles):
        info['稳定性'] = '不稳定 (存在右半平面极点)'
    else:
        info['稳定性'] = '临界稳定 (极点位于虚轴上)'

    den_coeffs = tf.den[0][0]
    if len(den_coeffs) == 3:
        a2, a1, a0 = den_coeffs
        info['ωn'] = np.sqrt(abs(a0 / a2))
        info['ζ'] = a1 / a2 / (2 * info['ωn'])
        info['ωn'] = np.sqrt(abs(a0 / a2))

    return info


def mason_gain(nodes, edges, input_node, output_node):
    """梅森增益公式（符号计算）"""
    import sympy as sp
    s = sp.Symbol('s')

    n = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}

    A = sp.zeros(n, n)
    for from_n, to_n, gain_expr in edges:
        i = node_idx[from_n]
        j = node_idx[to_n]
        try:
            g = sp.sympify(gain_expr)
        except Exception:
            g = sp.sympify(str(gain_expr))
        A[j, i] = g

    b = sp.zeros(n, 1)
    b[node_idx[input_node]] = 1

    I = sp.eye(n)
    M = I - A
    x = M.inv() * b

    tf_sym = sp.simplify(x[node_idx[output_node]])
    steps = f"方程组: (I - A)x = b·u\nA = {A}\n传递函数 = {tf_sym}"

    return tf_sym, steps
