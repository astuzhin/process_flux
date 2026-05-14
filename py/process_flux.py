import numpy as np
import pandas as pd
import uproot as up
import ROOT
import matplotlib.pyplot as plt

import mypy.constants as const
import mypy.pyroot_utils as pyus
import mypy.pyspline as pyspl
import mypy.pyroot_utils as pyus
from mypy.process import ProcessorData, ProcessorMC, broken_power_low_energy_log, acc_spline, calc_eff_weighted

import sys

def calc_spline(gr, xmin=None, xmax=None):
    gr = gr.dropna()
    x = gr.index.get_level_values('rig').mid
    xmin = x.min() if xmin is None else xmin
    xmax = x.max() if xmax is None else xmax
    if len(x[(x > xmin) & (x < xmax)]) < 5:
        return None
    model = pyspl.fit_spline(x, gr['corr_avg2mc'], gr['corr_avg2mc_err'], mode="log-lin", extrapolation=['tangent', 'constant'], lam=5, x_low=xmin, x_high=xmax)
    gr['corr_avg2mc_spl'] = pyspl.eval_spline(model, x)
    return gr

proc_data = ProcessorData(str(const.ISS_PATH), prefix="_last")
proc_data.load()
proc_mc = ProcessorMC(str(const.MC_PATH))

time = pd.Timestamp(sys.argv[1], tz="UTC")

cr = proc_data.count_rate.loc[time, ["cr", "cr_err"]]
eff = proc_data.effs_daily.xs(time, level='time')

cr = cr.dropna()
cr = cr[cr.index.mid < 100]

flux_model_x = cr.index.mid.to_numpy(copy=True)
flux_model_y = cr['cr'].to_numpy(copy=True)
flux_model_y_err = cr['cr_err'].to_numpy(copy=True)

def unfold_iter(cr, eff, flux_model_x, flux_model_y, flux_model_y_err):
    proc_mc.reweight(flux_model_x, flux_model_y, flux_model_y_err)
    eff['corr_avg2mc'] = eff['eff_avg'] / proc_mc.effs_mc['eff']
    eff['corr_avg2mc_err'] = ((eff['eff_err_avg'] / eff['eff_avg']) ** 2 + (proc_mc.effs_mc['eff_err'] / proc_mc.effs_mc['eff']) ** 2) ** 0.5
    eff = eff.groupby(['det', 'time_avg'], group_keys=False).apply(calc_spline)
    eff['corr'] = eff['corr_avg2mc_spl'] * eff['eff_daily2avg_spl']

    fx = cr.copy()

    fx['corr'] = eff['corr'].unstack('det').prod(axis=1)
    fx['acc'] = acc_spline(fx.index.mid, proc_mc.acc.index.mid, proc_mc.acc['eff'].values)
    fx['acc_gen'] = acc_spline(fx.index.mid, proc_mc.acc_gen.index.mid, proc_mc.acc_gen['eff'].values)
    fx['fx'] = fx['cr'] / (fx['acc'] * fx['corr'])
    fx['fx_err'] = fx['cr_err'] / (fx['acc'] * fx['corr'])

    return fx

print(f"Iteration 0")
fx = unfold_iter(cr, eff, flux_model_x, flux_model_y, flux_model_y_err)
for i in range(1, 10):
    print(f"Iteration {i}")
    flux_model_x = fx.index.mid.to_numpy(copy=True)
    flux_model_y = fx['fx'].to_numpy(copy=True)
    flux_model_y_err = fx['fx_err'].to_numpy(copy=True)
    fx_new = unfold_iter(cr, eff, flux_model_x, flux_model_y, flux_model_y_err)
    r = np.abs(fx_new['fx'] / fx['fx'] - 1)
    if np.all(r < 0.005):
        break
    fx = fx_new

print(fx)