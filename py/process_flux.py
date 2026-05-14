import numpy as np
import pandas as pd
import uproot as up
import ROOT
import matplotlib.pyplot as plt
import gc

from concurrent.futures import ProcessPoolExecutor, as_completed

import mypy.constants as const
import mypy.pyroot_utils as pyus
import mypy.pyspline as pyspl
import mypy.pyroot_utils as pyus
import mypy.process as proc
from mypy.process import ProcessorData, ProcessorMC, broken_power_low_energy_log, acc_spline, calc_eff_weighted

from numba import njit
from scipy.optimize import curve_fit
import sys

import os
import time
import psutil
import threading
def monitor_memory(interval=5):
    proc = psutil.Process(os.getpid())
    while True:
        procs = [proc] + proc.children(recursive=True)
        rss = 0
        for p in procs:
            try:
                rss += p.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        print(f"Total RSS: {rss / 1024**3:.2f} GB | processes: {len(procs)}")
        time.sleep(interval)

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

def unfold_iter(cr, eff, flux_model_x, flux_model_y, flux_model_y_err, rig_gen, mask, bins, acc_gen_t):
    eff_mc, acc = proc.reweight(rig_gen, mask, bins, acc_gen_t, flux_model_x, flux_model_y, flux_model_y_err)
    eff['corr_avg2mc'] = eff['eff_avg'] / eff_mc['eff']
    eff['corr_avg2mc_err'] = ((eff['eff_err_avg'] / eff['eff_avg']) ** 2 + (eff_mc['eff_err'] / eff_mc['eff']) ** 2) ** 0.5
    eff = eff.groupby(['det', 'time_avg'], group_keys=False).apply(calc_spline)
    eff['corr'] = eff['corr_avg2mc_spl'] * eff['eff_daily2avg_spl']

    fx = cr.copy()

    fx['corr'] = eff['corr'].unstack('det').prod(axis=1)
    fx['acc'] = acc_spline(fx.index.mid, acc.index.mid, acc['eff'].values)
    # fx['acc_gen'] = acc_spline(fx.index.mid, acc_gen.index.mid, acc_gen['eff'].values)
    fx['fx'] = fx['cr'] / (fx['acc'] * fx['corr'])
    fx['fx_err'] = fx['cr_err'] / (fx['acc'] * fx['corr'])

    return fx

from multiprocessing import shared_memory
SHARED = {}
def make_shared_array(arr):
    arr = np.asarray(arr)
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[:] = arr
    meta = {
        "name": shm.name,
        "shape": arr.shape,
        "dtype": arr.dtype.str,
    }
    return shm, meta

def attach_shared_arrays(shared_meta):
    global SHARED

    for key, meta in shared_meta.items():
        shm = shared_memory.SharedMemory(name=meta["name"])

        arr = np.ndarray(
            meta["shape"],
            dtype=np.dtype(meta["dtype"]),
            buffer=shm.buf,
        )

        SHARED[key] = arr
        SHARED[key + "_shm"] = shm

PROC_DATA = None
def init_worker(shared_meta):
    global PROC_DATA
    PROC_DATA = ProcessorData(str(const.ISS_PATH), prefix="_last")
    PROC_DATA.load()

    attach_shared_arrays(shared_meta)
    SHARED["bins"] = {
        "in": SHARED["bins_in"],
        "tf": SHARED["bins_tf"],
        "l1": SHARED["bins_l1"],
        "tr": SHARED["bins_tr"],
        "acc": SHARED["bins_acc"],
    }
    
def process_day(t):
    global PROC_DATA, SHARED
    print(f"Processing {t}")
    try:
        cr = PROC_DATA.count_rate.loc[t, ["cr", "cr_err"]]
        eff = PROC_DATA.effs_daily.xs(t, level='time')
    except KeyError:
        print(f"No data for {t}")
        return None

    cr = cr.dropna()
    cr = cr[cr.index.mid < 100]

    flux_model_x = cr.index.mid.to_numpy(copy=True)
    flux_model_y = cr['cr'].to_numpy(copy=True)
    flux_model_y_err = cr['cr_err'].to_numpy(copy=True)

    # print(f"Iteration 0")
    fx = unfold_iter(cr, eff, flux_model_x, flux_model_y, flux_model_y_err, SHARED['rig_gen'], SHARED['mask'], SHARED['bins'], SHARED['acc_gen_t'])
    for i in range(1, 10):
        # print(f"Iteration {i}")
        flux_model_x = fx.index.mid.to_numpy(copy=True)
        flux_model_y = fx['fx'].to_numpy(copy=True)
        flux_model_y_err = fx['fx_err'].to_numpy(copy=True)
        fx_new = unfold_iter(cr, eff, flux_model_x, flux_model_y, flux_model_y_err, SHARED['rig_gen'], SHARED['mask'], SHARED['bins'], SHARED['acc_gen_t'])
        r = np.abs(fx_new['fx'] / fx['fx'] - 1)
        if np.all(r < 0.005):
            break
        fx = fx_new

    print(fx)
    return fx

if __name__ == "__main__":
    threading.Thread(target=monitor_memory, daemon=True).start()

    file_mc = up.open(str(const.MC_PATH))
    tree = file_mc['eff_tree']
    ngen = file_mc['mc_info_tree'].arrays(library="np")['Ngen'].sum()
    rig = tree['rig'].array(library="np")
    rig_gen = tree['rig_gen'].array(library="np")
    rig_beta = tree['rig_beta'].array(library="np")
    rig_inn = tree['rig_inn'].array(library="np")
    rig_inl1 = tree['rig_inl1'].array(library="np")
    mask = tree['mask'].array(library="np")

    shared_objects = {
        "rig_gen": rig_gen,
        "mask": mask,
        "bins_in": np.digitize(np.where(rig_beta < 5.0, rig_beta, rig_gen), const.RBINS_EDGES).astype(np.uint16),
        "bins_tf": np.digitize(rig_inl1, const.RBINS_EDGES).astype(np.uint16),
        "bins_l1": np.digitize(rig_inn, const.RBINS_EDGES).astype(np.uint16),
        "bins_tr": np.digitize(rig, const.RBINS_EDGES).astype(np.uint16),
        "bins_acc": np.digitize(rig, const.RBINS_ACC_EDGES).astype(np.uint16),
        "acc_gen_t": ngen * np.log(const.RBINS_ACC_EDGES[1:] / const.RBINS_ACC_EDGES[:-1]) / np.log(const.RIG_MAX / const.RIG_MIN),
    }

    shms = []
    shared_meta = {}

    for key, arr in shared_objects.items():
        shm, meta = make_shared_array(arr)
        shms.append(shm)
        shared_meta[key] = meta
    
    del rig, rig_gen, rig_beta, rig_inn, rig_inl1, mask, shared_objects
    gc.collect()
    file_mc.close()

    t1 = pd.Timestamp(sys.argv[1], tz="UTC")
    t2 = pd.Timestamp(sys.argv[2], tz="UTC")
    tt = pd.date_range(t1, t2, freq="D")

    results = []

    executor = None
    try:
        executor = ProcessPoolExecutor(max_workers=4, initializer=init_worker, initargs=(shared_meta,))
        futures = {executor.submit(process_day, t): t for t in tt}
        for future in as_completed(futures):
            t = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error for {t}: {e}")
    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping workers...")
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

        for shm in shms:
            shm.close()
            shm.unlink()