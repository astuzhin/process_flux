import numpy as np

def calc_eff_weighted(p, t, p_w2, t_w2):
    p = np.asarray(p)
    t = np.asarray(t)
    p_w2 = np.asarray(p_w2)
    t_w2 = np.asarray(t_w2)
    eff = np.divide(p, t, out=np.full_like(p, np.nan, dtype=float), where=t != 0)
    var = np.divide(p_w2 * (1.0 - 2.0 * eff) + t_w2 * eff * eff, t * t, out=np.zeros_like(p, dtype=float), where=t != 0)
    var = np.maximum(var, 0.0)
    err = np.sqrt(var)
    return eff, err