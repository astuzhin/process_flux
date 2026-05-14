import numpy as np
from dataclasses import dataclass
from scipy.interpolate import make_smoothing_spline


@dataclass
class SplineModel:
    spl: callable
    mode: str
    x_low: float
    x_high: float
    extrapolation: tuple[str, str]  # (left x < x_low, right x > x_high)


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

_EXTRAP_ALIASES = {"constrant": "constant"}


def _canonical_extrap(name: str) -> str:
    key = name.strip().lower() if isinstance(name, str) else name
    key = _EXTRAP_ALIASES.get(key, key)
    if key not in ("constant", "tangent"):
        raise ValueError(
            f"Unknown extrapolation mode {name!r}; expected 'constant' or 'tangent'"
        )
    return key


def _normalize_extrapolation(extrapolation) -> tuple[str, str]:
    if isinstance(extrapolation, str):
        c = _canonical_extrap(extrapolation)
        return (c, c)
    if isinstance(extrapolation, (list, tuple)) and len(extrapolation) == 2:
        return (_canonical_extrap(extrapolation[0]), _canonical_extrap(extrapolation[1]))
    raise TypeError(
        "extrapolation must be a str or a sequence of two str, "
        f"got {type(extrapolation).__name__!r}"
    )

def _transform_x(x, mode):
    if mode in ("log-lin", "log-log"):
        return np.log10(x)
    return x


def _transform_y(y, mode):
    if mode == "log-log":
        return np.log10(y)
    return y


def _inverse_y(y, mode):
    if mode == "log-log":
        return 10 ** y
    return y


# =========================
# FIT
# =========================

def fit_spline(
    x,
    y,
    yerr=None,
    x_low=None,
    x_high=None,
    lam=None,
    mode="log-lin",
    extrapolation="constant",
):
    x = np.asarray(x)
    y = np.asarray(y)

    if x_low is None:
        x_low = x.min()
    if x_high is None:
        x_high = x.max()

    mask = (x >= x_low) & (x <= x_high)

    if mode in ("log-lin", "log-log"):
        mask &= (x > 0)
    if mode == "log-log":
        mask &= (y > 0)

    x_fit = _transform_x(x[mask], mode)
    y_fit = _transform_y(y[mask], mode)

    if yerr is not None:
        w = 1.0 / yerr[mask]
    else:
        w = None

    spl = make_smoothing_spline(x_fit, y_fit, lam=lam, w=w)

    extrap_pair = _normalize_extrapolation(extrapolation)

    return SplineModel(
        spl=spl,
        mode=mode,
        x_low=x_low,
        x_high=x_high,
        extrapolation=extrap_pair,
    )


# =========================
# PREDICT
# =========================

def eval_spline(model: SplineModel, x):
    x = np.asarray(x)
    result = np.empty_like(x, dtype=float)

    logx = _transform_x(x, model.mode)

    mask_mid = (x >= model.x_low) & (x <= model.x_high)

    # --- внутри диапазона ---
    y_mid = model.spl(logx[mask_mid])
    result[mask_mid] = _inverse_y(y_mid, model.mode)

    # --- границы ---
    lx_low = _transform_x(model.x_low, model.mode)
    lx_high = _transform_x(model.x_high, model.mode)

    y_low = model.spl(lx_low)
    y_high = model.spl(lx_high)

    extrap_left, extrap_right = model.extrapolation
    d = (
        model.spl.derivative()
        if (extrap_left == "tangent" or extrap_right == "tangent")
        else None
    )

    # --- слева (x < x_low) ---
    mask_lo = x < model.x_low
    if np.any(mask_lo):
        if extrap_left == "constant":
            result[mask_lo] = _inverse_y(y_low, model.mode)
        else:
            slope_low = d(lx_low)
            dx_low = logx[mask_lo] - lx_low
            y_left = y_low + slope_low * dx_low
            result[mask_lo] = _inverse_y(y_left, model.mode)

    # --- справа (x > x_high) ---
    mask_hi = x > model.x_high
    if np.any(mask_hi):
        if extrap_right == "constant":
            result[mask_hi] = _inverse_y(y_high, model.mode)
        else:
            slope_high = d(lx_high)
            dx_high = logx[mask_hi] - lx_high
            y_right = y_high + slope_high * dx_high
            result[mask_hi] = _inverse_y(y_right, model.mode)

    return result