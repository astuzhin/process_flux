import uproot as up
import numpy as np
import pandas as pd
import shutil
from scipy.optimize import curve_fit

from . import pyroot_utils as pyus
from . import constants as const
from . import pyspline as pyspl
from . import utils as ut

class ProcessorData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.fileUPROOT = up.open(file_path)

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
    
    def recreate_efficiencies(self):
        self.load_efficiencies()
        self.effs['tr'] = (self.effs['tr'].reset_index('rig').groupby('rig').rolling("15D", center=True).sum().swaplevel().sort_index())
        self.effs_avg = {}
        self.effs_daily = {}

        def fit_linear(x, a, b):
            return a * x + b

        def fit_time(gr):
            x = gr.index.get_level_values('time').view("int64")
            y = gr['eff_daily2avg'].to_numpy()
            y_err = gr['eff_daily2avg_err'].to_numpy()

            mask = (np.isfinite(x) & np.isfinite(y) & np.isfinite(y_err) & (y_err>0))
            if mask.sum() < 5:
                gr['eff_daily2avg_timefit'] = np.nan
                return gr
            
            x0 = np.min(x[mask])
            p, m = curve_fit(fit_linear, x[mask] - x0, y[mask], sigma=y_err[mask], absolute_sigma=True, p0=(1.0, np.mean(y[mask])))
            gr['eff_daily2avg_timefit'] = fit_linear(x - x0, *p)
            return gr
        
        def fit_rig(gr, xmin=None, xmax=None):
            x = gr.index.get_level_values('rig').mid.to_numpy()
            y = gr['eff_daily2avg'].to_numpy()
            y_err = gr['eff_daily2avg_err'].to_numpy()

            xmin = x.min() if xmin is None else xmin
            xmax = x.max() if xmax is None else xmax
            mask = (np.isfinite(x) & (x>xmin) & (x<xmax) & np.isfinite(y) & np.isfinite(y_err) & (y_err>0))
            if mask.sum() < 5:
                gr['eff_daily2avg_rigfit'] = np.nan
                return gr

            model = pyspl.fit_spline(x[mask], y[mask], y_err[mask], mode="log-lin", extrapolation=['tangent', 'constant'], lam=5, x_low=xmin, x_high=xmax)
            gr['eff_daily2avg_rigfit'] = pyspl.eval_spline(model, x)
            return gr

        for key, gr in self.effs.items():
            tbins_avg = pd.cut(gr.index.get_level_values('time'), bins=const.AVG_PERIODS[key], right=False)
            
            # daily efficiencies
            eff, err = ut.calc_eff_weighted(gr['p'], gr['t'], gr['pw2'], gr['tw2'])
            df = pd.DataFrame({'eff_daily': eff.ravel(), 'eff_err_daily': err.ravel()}, index=gr.index)
            df['time_avg'] = tbins_avg

            # avg in each period
            gr['time_avg'] = tbins_avg
            df_avg = gr.groupby(['rig', 'time_avg'], group_keys=False, observed=True).sum()
            eff, err = ut.calc_eff_weighted(df_avg['p'], df_avg['t'], df_avg['pw2'], df_avg['tw2'])
            df_avg = pd.DataFrame({'eff_avg': eff.ravel(), 'eff_err_avg': err.ravel()}, index=df_avg.index)

            df = df.reset_index().merge(df_avg, on=['rig', 'time_avg'], how='left').set_index(['rig', 'time'])

            df['eff_daily2avg'] = df['eff_daily'] / df['eff_avg']
            df['eff_daily2avg_err'] = ((df['eff_err_daily'] / df['eff_daily'])**2 + (df['eff_err_avg'] / df['eff_avg'])**2)**0.5

            # fit with time
            df = df.groupby(['rig', 'time_avg'], group_keys=False, observed=True).apply(fit_time)
            
            # fit with rig
            df = df.groupby('time', group_keys=False, observed=True).apply(fit_rig, xmin=const.X_LIMS_DAY2AVG[key][0], xmax=const.X_LIMS_DAY2AVG[key][1])
            self.effs_daily[key] = df
            

            # # average efficiencies
            # df = gr.groupby('rig').sum()
            # eff, err = ut.calc_eff_weighted(df['p'], df['t'], df['pw2'], df['tw2'])
            # self.effs_avg[key] = pd.DataFrame({'eff': eff.ravel(), 'eff_err': err.ravel()}, index=df.index)
            
        self.effs_daily = pd.concat(self.effs_daily, names=['det']).sort_index()
        # self.effs_avg = pd.concat(self.effs_avg, names=['det']).sort_index()

    def save(self):
        print(f"Saving data to {const.OUTPUT_DIR}")
        shutil.rmtree(const.OUTPUT_DIR, ignore_errors=True)
        const.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(self.count_rate, f"{const.OUTPUT_DIR}/count_rate.pkl")
        pd.to_pickle(self.effs_daily, f"{const.OUTPUT_DIR}/effs_daily.pkl")
        # pd.to_pickle(self.effs_avg, f"{const.OUTPUT_DIR}/effs_avg.pkl")
    
    def load(self):
        if not const.OUTPUT_DIR.exists():
            raise FileNotFoundError(f"Output directory {const.OUTPUT_DIR} does not exist")
        self.count_rate = pd.read_pickle(f"{const.OUTPUT_DIR}/count_rate.pkl")
        self.effs_daily = pd.read_pickle(f"{const.OUTPUT_DIR}/effs_daily.pkl")
        # self.effs_avg = pd.read_pickle(f"{const.OUTPUT_DIR}/effs_avg.pkl")
        print(f"Loaded data from {const.OUTPUT_DIR}")