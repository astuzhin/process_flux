import uproot as up
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import curve_fit

from . import constants as const
from . import utils as ut

p0 =          [0.0, 20.0, 2.5, 1.0, 1.0]
bounds = ([-np.inf, 1.0, 0.0, 0.01, 0.2], 
          [np.inf, 80.0, 5.0, 20.0, 8.0])
def broken_power_low_energy_log(R, logA, Rb, gamma1, Rc, alpha):
    s = 0.5
    gamma2 = 2.7
    log_broken = (-gamma1 * np.log(R / Rb) - (gamma2 - gamma1) * s * np.log1p((R / Rb)**(1/s)))
    log_low = -(Rc / R)**alpha
    return logA + log_broken + log_low

@njit(cache=True)
def bplel_weights_numba(R, logA, Rb, gamma1, Rc, alpha, log_ratio, out):
    """Compute weights = exp(broken_power_low_energy_log(R, ...)) * R * log_ratio in-place.

    Equivalent to the numpy expression but in a single pass with zero
    N_mc-sized temporaries. Result is written into `out`.
    """
    s = 0.5
    gamma2 = 2.7
    inv_s = 1.0 / s
    coef = (gamma2 - gamma1) * s
    n = R.shape[0]
    for i in range(n):
        r = R[i]
        r_rb = r / Rb
        log_broken = -gamma1 * np.log(r_rb) - coef * np.log1p(r_rb ** inv_s)
        log_low = -(Rc / r) ** alpha
        out[i] = np.exp(logA + log_broken + log_low) * r * log_ratio
    
@njit
def fill_sums_numba(mask, bins_in, bins_tf, bins_l1, bins_tr, bins_acc, weights, n_rbins, n_abins):
    sums_pass = np.zeros((4, n_rbins), dtype=np.float64)
    sums_pass2 = np.zeros((4, n_rbins), dtype=np.float64)
    sums_total = np.zeros((4, n_rbins), dtype=np.float64)
    sums_total2 = np.zeros((4, n_rbins), dtype=np.float64)

    acc_pass = np.zeros(n_abins, dtype=np.float64)
    acc_pass2 = np.zeros(n_abins, dtype=np.float64)

    use_weights = len(weights) > 0

    for i in range(len(mask)):
        w = weights[i] if use_weights else 1.0
        w2 = w * w
        m = mask[i]

        # l1
        if m & const.L1_T:
            b = bins_l1[i]
            sums_total[const.L1, b] += w
            sums_total2[const.L1, b] += w2
            if m & const.L1_P:
                sums_pass[const.L1, b] += w
                sums_pass2[const.L1, b] += w2

        # tf
        if m & const.TF_T:
            b = bins_tf[i]
            sums_total[const.TF, b] += w
            sums_total2[const.TF, b] += w2
            if m & const.TF_P:
                sums_pass[const.TF, b] += w
                sums_pass2[const.TF, b] += w2

        # inn
        if m & const.IN_T:
            b = bins_in[i]
            sums_total[const.IN, b] += w
            sums_total2[const.IN, b] += w2
            if m & const.IN_P:
                sums_pass[const.IN, b] += w
                sums_pass2[const.IN, b] += w2

        pass_phys = (m & const.MAIN_PHYS) != 0
        pass_unb = (m & const.MAIN_UNB) != 0

        # tr
        if pass_phys or pass_unb:
            b = bins_tr[i]
            sums_total[const.TR, b] += w
            sums_total2[const.TR, b] += w2
            if pass_phys:
                sums_pass[const.TR, b] += w
                sums_pass2[const.TR, b] += w2

        # acc
        if pass_phys:
            b = bins_acc[i]
            acc_pass[b] += w
            acc_pass2[b] += w2

    return sums_pass[:, 1:-1], sums_pass2[:, 1:-1], sums_total[:, 1:-1], sums_total2[:, 1:-1], acc_pass[1:-1], acc_pass2[1:-1]


def acc_spline(x_new, x, y):
        lx = np.log(x)
        y_smooth = pd.Series(y, index=lx).rolling(window=75, center=True, min_periods=1).mean()
        y_new = np.interp(np.log(x_new), lx, y_smooth)
        return y_new

def calc_weights(rig_gen, flux_model_x, flux_model_y, flux_model_y_err, weights_out=None):
    """Fit a broken-power-low model to the (x,y) flux and compute MC reweighting weights.

    If `weights_out` is provided, the N_mc weights are written into it in-place
    (no allocation). Otherwise a fresh array is allocated. `weights_out` must
    have shape (len(rig_gen),) and dtype float64.
    """
    norm = np.trapezoid(flux_model_y, flux_model_x)
    flux_model_y /= norm
    flux_model_y_err /= norm
    p0[0] = np.log(np.max(flux_model_y))
    popt, pcov = curve_fit(broken_power_low_energy_log, flux_model_x, np.log(flux_model_y), 
        p0=p0, sigma=flux_model_y_err / flux_model_y, absolute_sigma=True, maxfev=100000, bounds=bounds)

    log_ratio = np.log(const.RIG_MAX / const.RIG_MIN)

    if weights_out is None:
        weights_out = np.empty(rig_gen.shape[0], dtype=np.float64)
    bplel_weights_numba(rig_gen, *popt, log_ratio, weights_out)

    x_acc = const.RBINS_ACC.mid.to_numpy(copy=True)
    weights_acc = np.exp(broken_power_low_energy_log(x_acc, *popt))
    weights_acc *= x_acc
    weights_acc *= log_ratio
    return weights_out, weights_acc

def reweight(rig_gen, mask, bins, acc_gen_t, flux_model_x, flux_model_y, flux_model_y_err, weights_out=None):
    weights, weights_acc = calc_weights(rig_gen, flux_model_x, flux_model_y, flux_model_y_err, weights_out=weights_out)
    p, p_w2, t, t_w2, acc_p, acc_p_w2 = fill_sums_numba(mask, bins["in"], bins["tf"], bins["l1"], bins["tr"], bins["acc"], 
                                                        weights, len(const.RBINS_EDGES) + 1, len(const.RBINS_ACC_EDGES) + 1)
    eff, err = ut.calc_eff_weighted(p, t, p_w2, t_w2)
    effs_mc = pd.DataFrame({'eff': eff.ravel(), 'eff_err': err.ravel()}, index=const.EFF_IDX)
    acc_t = acc_gen_t * weights_acc
    acc_t_w2 = acc_gen_t * weights_acc ** 2
    eff, err = ut.calc_eff_weighted(acc_p, acc_t, acc_p_w2, acc_t_w2)
    acc = pd.DataFrame({'eff': eff.ravel(), 'err': err.ravel()}, index=const.RBINS_ACC) * np.pi * 3.9**2
    return effs_mc, acc


class ProcessorMC:
    def __init__(self, file_path):
        file = up.open(file_path)
        tree = file['eff_tree']
        
        self.ngen = file['mc_info_tree'].arrays(library="np")['Ngen'].sum()

        rig = tree['rig'].array(library="np")
        self.rig_gen = tree['rig_gen'].array(library="np")
        rig_beta = tree['rig_beta'].array(library="np")
        rig_inn = tree['rig_inn'].array(library="np")
        rig_inl1 = tree['rig_inl1'].array(library="np")
        self.mask = tree['mask'].array(library="np")

        self.acc_gen_p = np.histogram(self.rig_gen[(self.mask & const.MAIN_PHYS) != 0], bins=const.RBINS_ACC_EDGES)[0]
        self.acc_gen_t = self.ngen * np.log(const.RBINS_ACC_EDGES[1:] / const.RBINS_ACC_EDGES[:-1]) / np.log(const.RIG_MAX / const.RIG_MIN)
        eff, err = ut.calc_eff_weighted(self.acc_gen_p, self.acc_gen_t, self.acc_gen_p, self.acc_gen_t)
        self.acc_gen = pd.DataFrame({'eff': eff.ravel(), 'err': err.ravel()}, index=const.RBINS_ACC) * np.pi * 3.9**2

        self.bins = {
            "in": np.digitize(np.where(rig_beta < 5.0, rig_beta, self.rig_gen), const.RBINS_EDGES).astype(np.uint16),
            "l1": np.digitize(rig_inn, const.RBINS_EDGES).astype(np.uint16),
            "tf": np.digitize(rig_inl1, const.RBINS_EDGES).astype(np.uint16),
            "tr": np.digitize(rig, const.RBINS_EDGES).astype(np.uint16),
            "acc": np.digitize(rig, const.RBINS_ACC_EDGES).astype(np.uint16)
        }

        self._eff_idx = pd.MultiIndex.from_product([const.DETS.keys(), const.RBINS], names=['det', 'rig'])