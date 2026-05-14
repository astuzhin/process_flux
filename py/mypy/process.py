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

def calc_eff_weighted(p, t, p_w2, t_w2, index=None):
    p = np.asarray(p)
    t = np.asarray(t)
    p_w2 = np.asarray(p_w2)
    t_w2 = np.asarray(t_w2)
    eff = np.divide(p, t, out=np.full_like(p, np.nan, dtype=float), where=t != 0)
    var = np.divide(p_w2 * (1.0 - 2.0 * eff) + t_w2 * eff * eff, t * t, out=np.zeros_like(p, dtype=float), where=t != 0)
    var = np.maximum(var, 0.0)
    err = np.sqrt(var)
    cols = {"eff": eff, "eff_err": err}
    return pd.DataFrame(cols) if index is None else pd.DataFrame(cols, index=index)

p0 =          [0.0, 20.0, 2.5, 1.0, 1.0]
bounds = ([-np.inf, 1.0, 0.0, 0.01, 0.2], 
          [np.inf, 80.0, 5.0, 20.0, 8.0])
def broken_power_low_energy_log(R, logA, Rb, gamma1, Rc, alpha):
    s = 0.5
    gamma2 = 2.7
    log_broken = (-gamma1 * np.log(R / Rb) - (gamma2 - gamma1) * s * np.log1p((R / Rb)**(1/s)))
    log_low = -(Rc / R)**alpha
    return logA + log_broken + log_low

def acc_spline(x_new, x, y):
        lx = np.log(x)
        y_smooth = pd.Series(y, index=lx).rolling(window=75, center=True, min_periods=1).mean()
        y_new = np.interp(np.log(x_new), lx, y_smooth)
        return y_new

class ProcessorMC:
    def __init__(self, file_path):
        self.init_root_cpp()

        self.file_path = file_path
        self.file_mc = up.open(file_path)
        self.mc_rw = ROOT.McRW(str(file_path))
        self.mc_rw.set_rbins(const.RBINS_EDGES, const.RBINS_ACC_EDGES)

        self.mc_rw.load_tree()
    
    def init_root_cpp(self):
        ROOT.gErrorIgnoreLevel = ROOT.kError
        inc = const.INCLUDE_DIR.resolve().as_posix()
        bld = const.BUILD_DIR.resolve().as_posix()

        ext = ".dylib" if platform.system() == "Darwin" else ".so"
        lib = f"{bld}/libprocess_flux{ext}"

        headers = (
            "McRW.hpp",
        )
        for h in headers:
            ROOT.gInterpreter.ProcessLine(f'#include "{inc}/{h}"')
        ROOT.gSystem.Load(str(lib))

    def calc_acc_gen(self):
        n_gen = self.file_mc["mc_info_tree"].arrays(library="np")['Ngen'].sum()
        tree = self.file_mc["eff_tree"]
        self.rig_gen = tree['rig_gen'].array(library="np")
        mask = tree['mask'].array(library="np")
        pass_phys = (mask & (1 << ROOT.Cuts.main_phys)) != 0
        acc_gen_p = np.histogram(self.rig_gen[pass_phys], bins=const.RBINS_ACC_EDGES)[0]
        self.acc_gen_t = n_gen * np.log(const.RBINS_ACC_EDGES[1:] / const.RBINS_ACC_EDGES[:-1]) / np.log(const.RIG_MAX / const.RIG_MIN)
        self.acc_gen = calc_eff_weighted(acc_gen_p, self.acc_gen_t, acc_gen_p, self.acc_gen_t, index=const.RBINS_ACC) * np.pi * 3.9**2
    
    def reweight(self, flux_model_x, flux_model_y, flux_model_y_err):
        norm = np.trapezoid(flux_model_y, flux_model_x)
        flux_model_y /= norm
        flux_model_y_err /= norm
        p0[0] = np.log(np.max(flux_model_y))
        popt, pcov = curve_fit(broken_power_low_energy_log, flux_model_x, np.log(flux_model_y), 
            p0=p0, sigma=flux_model_y_err / flux_model_y, absolute_sigma=True, maxfev=100000, bounds=bounds)
        self.popt = popt
        weights = np.exp(broken_power_low_energy_log(self.rig_gen, *popt))
        weights *= (self.rig_gen * np.log(const.RIG_MAX / const.RIG_MIN))
        weights_acc = np.exp(broken_power_low_energy_log(const.RBINS_ACC.mid, *popt))
        weights_acc *= (const.RBINS_ACC.mid * np.log(const.RIG_MAX / const.RIG_MIN))
        self.mc_rw.fill_sums(weights)

        df = {}
        for det in ROOT.DetList:
            if det == ROOT.Det.acc:
                continue
            p, p_w2 = self.mc_rw.get_sums_pass()[det]
            t, t_w2 = self.mc_rw.get_sums_total()[det]
            gr = calc_eff_weighted(p, t, p_w2, t_w2)
            gr = gr.iloc[1:-1]
            gr.index = const.RBINS
            df[ROOT.DetNames[det]] = gr
        df = pd.concat(df)
        df.index.names = ["det", "rig"]

        self.effs_mc = df

        acc_p = np.array(self.mc_rw.get_sums_pass()[ROOT.Det.acc][0])[1:-1]
        acc_p_w2 = np.array(self.mc_rw.get_sums_pass()[ROOT.Det.acc][1])[1:-1]
        acc_t = self.acc_gen_t * weights_acc
        acc_t_w2 = self.acc_gen_t * weights_acc ** 2
        self.acc = calc_eff_weighted(acc_p, acc_t, acc_p_w2, acc_t_w2, index=const.RBINS_ACC) * np.pi * 3.9**2

    def reweight_old(self, cr=None):
        if cr is not None:
            cr = cr[cr.index.mid < 100].dropna()
            norm = np.trapezoid(cr['cr'], cr.index.mid)
            cr = cr[['cr', 'cr_err']] / norm
            p0 = [np.log(np.max(cr['cr'])), 20.0, 2.5, 2.7, 1.0, 1.0]
            bounds = ([-np.inf, 1.0, 0.0, 0.01, 0.2], 
                      [np.inf, 80.0, 5.0, 20.0, 8.0])
            self.cr_pdf = cr
            popt, pcov = curve_fit(broken_power_low_energy_log, cr.index.mid, np.log(cr['cr']), 
                p0=p0, sigma=cr['cr_err'] / cr['cr'], absolute_sigma=True, maxfev=100000)
            # model_cr = pyspl.fit_spline(
            #     cr.index.mid, cr['cr'] / norm, cr['cr_err'] / norm, 
            #     mode="log-log", extrapolation="tangent", 
            #     lam=0.5, x_low=cr.index.mid.min(), x_high=100)
            # weights = pyspl.eval_spline(model_cr, self.rig_gen) * self.rig_gen * np.log(const.RIG_MAX / const.RIG_MIN)
            # self.model_cr = model_cr
            self.popt = popt
            weights = np.exp(broken_power_low_energy_log(self.rig_gen, *popt))
            weights *= (self.rig_gen * np.log(const.RIG_MAX / const.RIG_MIN))
            weights_acc = np.exp(broken_power_low_energy_log(const.RBINS_ACC.mid, *popt))
            weights_acc *= (const.RBINS_ACC.mid * np.log(const.RIG_MAX / const.RIG_MIN))
            self.mc_rw.fill_sums(weights)
        else:
            self.mc_rw.fill_sums()
            weights_acc = 1

        df = {}
        for det in ROOT.DetList:
            if det == ROOT.Det.acc:
                continue
            p, p_w2 = self.mc_rw.get_sums_pass()[det]
            t, t_w2 = self.mc_rw.get_sums_total()[det]
            gr = calc_eff_weighted(p, t, p_w2, t_w2)
            gr = gr.iloc[1:-1]
            gr.index = const.RBINS
            df[ROOT.DetNames[det]] = gr
        df = pd.concat(df)
        df.index.names = ["det", "rig"]

        self.effs_mc = df

        acc_p = np.array(self.mc_rw.get_sums_pass()[ROOT.Det.acc][0])[1:-1]
        acc_p_w2 = np.array(self.mc_rw.get_sums_pass()[ROOT.Det.acc][1])[1:-1]
        acc_t = self.acc_gen_t * weights_acc
        acc_t_w2 = self.acc_gen_t * weights_acc ** 2
        self.acc = calc_eff_weighted(acc_p, acc_t, acc_p_w2, acc_t_w2, index=const.RBINS_ACC) * np.pi * 3.9**2

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
            gr = gr.dropna()
            x = gr.index.get_level_values('rig').mid
            xmin = x.min() if xmin is None else xmin
            xmax = x.max() if xmax is None else xmax
            if len(x[(x > xmin) & (x < xmax)]) < 5:
                return None
            model = pyspl.fit_spline(x, gr['eff_daily2avg'], gr['eff_daily2avg_err'], mode="log-lin", extrapolation=['tangent', 'constant'], lam=5, x_low=xmin, x_high=xmax)
            gr['eff_daily2avg_spl'] = pyspl.eval_spline(model, x)
            return gr
        
        for key, gr in self.effs.items():
            tbins_avg = pd.cut(gr.index.get_level_values('time'), bins=const.AVG_PERIODS[key], right=False)

            # daily efficiencies
            df_daily = calc_eff_weighted(gr['p'], gr['t'], gr['pw2'], gr['tw2'], index=gr.index)
            df_daily['time_avg'] = tbins_avg

            # average efficiencies
            df = gr.groupby([tbins_avg, 'rig']).sum()
            df_avg = calc_eff_weighted(df['p'], df['t'], df['pw2'], df['tw2'], index=df.index)
            df_avg.index.names = ['time_avg', 'rig']
            self.effs_avg[key] = df_avg

            df_daily = df_daily.reset_index().merge(df_avg.reset_index(), on=['time_avg', 'rig'], how='left', suffixes=('_daily', '_avg')).dropna().set_index(['time', 'rig']).sort_index()
            df_daily['eff_daily2avg'] = df_daily['eff_daily'] / df_daily['eff_avg']
            df_daily['eff_daily2avg_err'] = ((df_daily['eff_err_daily'] / df_daily['eff_daily']) ** 2) ** 0.5

            df_daily = df_daily.groupby('time', group_keys=False).apply(calc_spline, xmin=const.X_LIMS[key][0], xmax=const.X_LIMS[key][1])
            
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


class Processor:
    def __init__(self):
        self.path_iss = const.ISS_PATH
        self.path_mc = const.MC_PATH