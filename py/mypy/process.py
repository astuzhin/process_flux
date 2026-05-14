from typing import Any
import numpy as np
import pandas as pd
import platform
import ROOT
import uproot as up

from . import pyroot_utils as pyus
from . import pyspline as pyspl
from . import constants as const

from scipy.optimize import curve_fit

from numba import njit

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
    eff, err = calc_eff_weighted(p, t, p_w2, t_w2)
    effs_mc = pd.DataFrame({'eff': eff.ravel(), 'eff_err': err.ravel()}, index=const.EFF_IDX)
    acc_t = acc_gen_t * weights_acc
    acc_t_w2 = acc_gen_t * weights_acc ** 2
    eff, err = calc_eff_weighted(acc_p, acc_t, acc_p_w2, acc_t_w2)
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
        eff, err = calc_eff_weighted(self.acc_gen_p, self.acc_gen_t, self.acc_gen_p, self.acc_gen_t)
        self.acc_gen = pd.DataFrame({'eff': eff.ravel(), 'err': err.ravel()}, index=const.RBINS_ACC) * np.pi * 3.9**2

        self.bins = {
            "in": np.digitize(np.where(rig_beta < 5.0, rig_beta, self.rig_gen), const.RBINS_EDGES).astype(np.uint16),
            "l1": np.digitize(rig_inn, const.RBINS_EDGES).astype(np.uint16),
            "tf": np.digitize(rig_inl1, const.RBINS_EDGES).astype(np.uint16),
            "tr": np.digitize(rig, const.RBINS_EDGES).astype(np.uint16),
            "acc": np.digitize(rig, const.RBINS_ACC_EDGES).astype(np.uint16)
        }

        self._eff_idx = pd.MultiIndex.from_product([const.DETS.keys(), const.RBINS], names=['det', 'rig'])

class ProcessorData:
    def __init__(self, file_path, prefix=""):
        self.file_path = file_path
        self.fileUPROOT = up.open(file_path)

        self.prefix = prefix

    def init_count_rate(self):
        lv = self.fileUPROOT["livetime"]
        counts = self.fileUPROOT["tr_p"]
        lv = pyus.UPROOT.th_to_df(lv, bin_names=['time', 'rig'], col_names=['lv', 'sumw2'], is_interval=[False, True], is_time=[True, False])
        counts = pyus.UPROOT.th_to_df(counts, bin_names=['time', 'rig'], col_names=['counts', 'sumw2'], is_interval=[False, True], is_time=[True, False])
        cr = pd.concat([counts['counts'], lv['lv']], axis=1)
        rbins = cr.index.get_level_values("rig")
        cr['dRig'] = rbins.right - rbins.left
        cr['cr'] = cr['counts'] / cr['lv'] / cr['dRig']
        cr['cr_err'] = np.sqrt(cr['counts']) / cr['lv'] / cr['dRig']
        time = cr.index.get_level_values('time')
        mask = (time >= const.TMIN_GLOB) & (time <= const.TMAX_GLOB)
        self.count_rate = cr[mask]
    
    def load_efficiencies(self):
        df = {}
        for det in ['tr', 'in', 'tf', 'l1']:
            p = self.fileUPROOT[f"{det}_p"]
            t = self.fileUPROOT[f"{det}_t"]
            p = pyus.UPROOT.th_to_df(p, bin_names=['time', 'rig'], col_names=['p', 'pw2'], is_interval=[False, True], is_time=[True, False])
            t = pyus.UPROOT.th_to_df(t, bin_names=['time', 'rig'], col_names=['t', 'tw2'], is_interval=[False, True], is_time=[True, False])
            gr = pd.concat([p, t], axis=1)
            df[det] = gr
        self.effs = df
    
    def calc_efficiencies(self):
        self.load_efficiencies()

        self.effs['tr'] = (self.effs['tr'].reset_index('rig').groupby('rig').rolling("14D", center=True).sum().swaplevel().sort_index())
        self.effs_avg = {}
        self.effs_daily = {}

        def calc_spline(gr, xmin=None, xmax=None):
            gr_filtered = gr.dropna()
            x = gr_filtered.index.get_level_values('rig').mid
            xmin = x.min() if xmin is None else xmin
            xmax = x.max() if xmax is None else xmax
            if len(x[(x > xmin) & (x < xmax)]) < 5:
                return None
            model = pyspl.fit_spline(x, gr_filtered['eff_daily2avg'], gr_filtered['eff_daily2avg_err'], mode="log-lin", extrapolation=['tangent', 'constant'], lam=5, x_low=xmin, x_high=xmax)
            gr['eff_daily2avg_spl'] = pyspl.eval_spline(model, gr.index.get_level_values('rig').mid)
            return gr
        
        for key, gr in self.effs.items():
            tbins_avg = pd.cut(gr.index.get_level_values('time'), bins=const.AVG_PERIODS[key], right=False)

            # daily efficiencies
            eff, err = calc_eff_weighted(gr['p'], gr['t'], gr['pw2'], gr['tw2'])
            df_daily = pd.DataFrame({'eff_daily': eff.ravel(), 'eff_err_daily': err.ravel()}, index=gr.index)
            df_daily['time_avg'] = tbins_avg

            # average efficiencies
            df = gr.groupby([tbins_avg, 'rig']).sum()
            eff, err = calc_eff_weighted(df['p'], df['t'], df['pw2'], df['tw2'])
            df_avg = pd.DataFrame({'eff_avg': eff.ravel(), 'eff_err_avg': err.ravel()}, index=df.index)
            df_avg.index.names = ['time_avg', 'rig']
            self.effs_avg[key] = df_avg

            df_daily = df_daily.reset_index().merge(df_avg.reset_index(), on=['time_avg', 'rig'], how='left', suffixes=('_daily', '_avg')).dropna().set_index(['time', 'rig']).sort_index()
            df_daily['eff_daily2avg'] = df_daily['eff_daily'] / df_daily['eff_avg']
            df_daily['eff_daily2avg_err'] = ((df_daily['eff_err_daily'] / df_daily['eff_daily']) ** 2 + (df_daily['eff_err_avg'] / df_daily['eff_avg']) ** 2) ** 0.5

            df_daily = df_daily.groupby('time', group_keys=False).apply(calc_spline, xmin=const.X_LIMS_DAY2AVG[key][0], xmax=const.X_LIMS_DAY2AVG[key][1])
            
            self.effs_daily[key] = df_daily

        self.effs_avg = pd.concat(self.effs_avg, names=['det']).sort_index()
        self.effs_daily = pd.concat(self.effs_daily, names=['det']).sort_index()

    def save(self):
        pd.to_pickle(self.count_rate, f"{const.DATA_DIR}/count_rate{self.prefix}.pkl")
        pd.to_pickle(self.effs_daily, f"{const.DATA_DIR}/effs_daily{self.prefix}.pkl")
        pd.to_pickle(self.effs_avg, f"{const.DATA_DIR}/effs_avg{self.prefix}.pkl")
    
    def load(self):
        self.count_rate = pd.read_pickle(f"{const.DATA_DIR}/count_rate{self.prefix}.pkl")
        self.effs_daily = pd.read_pickle(f"{const.DATA_DIR}/effs_daily{self.prefix}.pkl")
        self.effs_avg = pd.read_pickle(f"{const.DATA_DIR}/effs_avg{self.prefix}.pkl")

