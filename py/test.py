import numpy as np
import pandas as pd
import uproot as up
import ROOT
import matplotlib.pyplot as plt

import mypy.constants as const
import mypy.pyroot_utils as pyus
import mypy.pyspline as pyspl
import mypy.pyroot_utils as pyus
from mypy.process import ProcessorData, ProcessorMC, broken_power_low_energy_log, acc_spline

def calc_spline(gr):
    det = gr.name[0]
    gr = gr.dropna()
    x = gr.index.get_level_values('rig').mid
    # xmin = const.X_LIMS[det][0]
    # xmax = const.X_LIMS[det][1]
    model = pyspl.fit_spline(x, gr['corr_avg2mc'], gr['corr_avg2mc_err'], mode="log-lin", extrapolation=['tangent', 'constant'], lam=5, x_low=x.min(), x_high=x.max())
    gr['corr_avg2mc_spl'] = pyspl.eval_spline(model, x)
    return gr

def process_day(time):
    try:
        cr_day = proc_data.count_rate.loc[time, ["cr", "cr_err"]]
    except:
        print(f"Time {time} not found in count rate")
        return None

    try:
        eff_daily_day = proc_data.effs_daily.xs(time, level='time')
    except:
        print(f"Time {time} not found in efficiencies")
        return None

    cr_day = cr_day.dropna()
    cr_day = cr_day[cr_day.index.mid < 100]
    if len(cr_day) < 10:
        return None
    flux_model_x = cr_day.index.mid.to_numpy(copy=True)
    flux_model_y = cr_day['cr'].to_numpy(copy=True)
    flux_model_y_err = cr_day['cr_err'].to_numpy(copy=True)

    def unfold_iter(flux_model_x, flux_model_y, flux_model_y_err):
        fx = cr_day[['cr', 'cr_err']].copy()
        proc_mc.reweight(flux_model_x, flux_model_y, flux_model_y_err)
        eff = eff_daily_day.copy()

        eff['corr_avg2mc'] = eff['eff_avg'] / proc_mc.effs_mc['eff']
        eff['corr_avg2mc_err'] = ((eff['eff_err_avg'] / eff['eff_avg']) ** 2 + (proc_mc.effs_mc['eff_err'] / proc_mc.effs_mc['eff']) ** 2) ** 0.5

        eff = eff.groupby(['det', 'time_avg'], group_keys=False).apply(calc_spline)
        eff['corr'] = eff['corr_avg2mc_spl'] * eff['eff_daily2avg_spl']

        for det in const.DET_ORDER:
            fx[f'corr_daily2avg_spl_{det}'] = eff[f'eff_daily2avg_spl'].xs(det, level='det')
            fx[f'corr_avg2mc_spl_{det}'] = eff[f'corr_avg2mc_spl'].xs(det, level='det')
        fx['corr'] = eff['corr'].unstack('det').prod(axis=1)
        fx['acc'] = acc_spline(fx.index.mid, proc_mc.acc.index.mid, proc_mc.acc['eff'].values)
        fx['acc_gen'] = acc_spline(fx.index.mid, proc_mc.acc_gen.index.mid, proc_mc.acc_gen['eff'].values)
        fx['fx'] = fx['cr'] / (fx['acc'] * fx['corr'])
        fx['fx_err'] = fx['cr_err'] / (fx['acc'] * fx['corr'])
        return fx
    
    # print(f"Iteration 0")
    fx = unfold_iter(flux_model_x, flux_model_y, flux_model_y_err)
    for i in range(1, 10):
        # print(f"Iteration {i}")
        fx = fx.dropna()
        if len(fx) < 10:
            break
        flux_model_x = fx.index.mid.to_numpy(copy=True)
        flux_model_y = fx['fx'].to_numpy(copy=True)
        flux_model_y_err = fx['fx_err'].to_numpy(copy=True)
        fx_new = unfold_iter(flux_model_x, flux_model_y, flux_model_y_err)
        r = np.abs(fx_new['fx'] / fx['fx'] - 1)
        if np.all(r < 0.005):
            break
        fx = fx_new
    return fx


from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

proc_data = None
proc_mc = None
def init_worker():
    global proc_data, proc_mc
    proc_data = ProcessorData(str(const.ISS_PATH), prefix="_last")
    proc_data.load()
    proc_data.count_rate = proc_data.count_rate.loc["2012-01-01":"2025-01-01"]

    proc_mc = ProcessorMC(str(const.MC_PATH))
    proc_mc.calc_acc_gen()

def process(time):
    try:
        fx = process_day(time)
    except Exception as e:
        print(f"Error processing time {time}: {e}")
        return None
    return fx

import sys

if __name__ == "__main__":
    args = sys.argv[1:]

    proc_data = ProcessorData(str(const.ISS_PATH), prefix="_last")
    if args[0] == "recreate":
        print("Recreating count rate and efficiencies")
        proc_data.init_count_rate()
        proc_data.calc_efficiencies()
        proc_data.save()
        exit()

    ProcID = int(args[0])
    t1 = pd.Timestamp(args[1], tz="UTC")
    t2 = pd.Timestamp(args[2], tz="UTC")

    print(f"Processing {ProcID} from {t1} to {t2}")
    print("Loading count rate and efficiencies")
    proc_data.load()

    proc_data.count_rate = proc_data.count_rate.loc["2012-01-01":"2025-01-01"]
    times = proc_data.count_rate.index.get_level_values('time').unique()
    times = times[(times >= t1) & (times < t2)]

    data = {}
    with ProcessPoolExecutor(max_workers=4, initializer=init_worker) as executor:
        futures = {executor.submit(process, time): time for time in times}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing days"):
            time = futures[future]
            data[time] = future.result()
    data = pd.concat(data, names=['time']).sort_index()
    data.to_pickle(const.DATA_DIR / f"flux_{ProcID}.pkl")

