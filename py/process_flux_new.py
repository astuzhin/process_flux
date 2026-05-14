"""Daily cosmic-ray flux unfolding with iterative MC reweighting.

Usage:
    python process_flux_new.py <start_date> <end_date>

The script:
  1. Loads MC arrays and places them in shared memory (one copy for all workers).
  2. For each day in the requested range, launches an unfolding loop in a worker
     process: reweight MC to current flux estimate, recompute efficiencies and
     acceptance, derive a new flux, repeat until convergence.
"""

import gc
import os
import sys
import time
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from typing import Optional
from tqdm import tqdm

import numpy as np
import pandas as pd
import psutil
import uproot as up

import mypy.constants as const
import mypy.process as proc
import mypy.pyspline as pyspl
from mypy.process import ProcessorData, acc_spline


# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
NUM_WORKERS = 10
MAX_ITERATIONS = 10
CONVERGENCE_TOL = 0.005
RIG_MAX_USED = 100.0          # ignore count-rate bins above this rigidity (GV)
MEMORY_LOG_INTERVAL_S = 5.0


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------
def monitor_memory(interval: float = MEMORY_LOG_INTERVAL_S) -> None:
    """Periodically print the total RSS of this process and its children."""
    main_proc = psutil.Process(os.getpid())
    while True:
        procs = [main_proc] + main_proc.children(recursive=True)
        rss = 0
        for p in procs:
            try:
                rss += p.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        print(f"Total RSS: {rss / 1024**3:.2f} GB | processes: {len(procs)}")
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Shared-memory helpers
# ---------------------------------------------------------------------------
def make_shared_array(arr: np.ndarray) -> tuple[shared_memory.SharedMemory, dict]:
    """Copy `arr` into a new SharedMemory block; return the block and its metadata."""
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


def attach_shared_arrays(shared_meta: dict) -> dict:
    """Attach to SharedMemory blocks described by `shared_meta` in a worker."""
    shared: dict = {}
    for key, meta in shared_meta.items():
        shm = shared_memory.SharedMemory(name=meta["name"])
        arr = np.ndarray(
            meta["shape"],
            dtype=np.dtype(meta["dtype"]),
            buffer=shm.buf,
        )
        shared[key] = arr
        shared[key + "_shm"] = shm
    return shared


# ---------------------------------------------------------------------------
# MC loading
# ---------------------------------------------------------------------------
def load_mc(path: str) -> dict[str, np.ndarray]:
    """Read MC eff_tree from `path` and build the arrays shared with workers."""
    file_mc = up.open(path)
    try:
        tree = file_mc["eff_tree"]
        ngen = file_mc["mc_info_tree"].arrays(library="np")["Ngen"].sum()
        rig = tree["rig"].array(library="np")
        rig_gen = tree["rig_gen"].array(library="np")
        rig_beta = tree["rig_beta"].array(library="np")
        rig_inn = tree["rig_inn"].array(library="np")
        rig_inl1 = tree["rig_inl1"].array(library="np")
        mask = tree["mask"].array(library="np")
    finally:
        file_mc.close()

    log_ratio = np.log(const.RIG_MAX / const.RIG_MIN)
    return {
        "rig_gen": rig_gen,
        "mask": mask,
        "bins_in": np.digitize(np.where(rig_beta < 5.0, rig_beta, rig_gen), const.RBINS_EDGES).astype(np.uint16),
        "bins_tf": np.digitize(rig_inl1, const.RBINS_EDGES).astype(np.uint16),
        "bins_l1": np.digitize(rig_inn, const.RBINS_EDGES).astype(np.uint16),
        "bins_tr": np.digitize(rig, const.RBINS_EDGES).astype(np.uint16),
        "bins_acc": np.digitize(rig, const.RBINS_ACC_EDGES).astype(np.uint16),
        "acc_gen_t": ngen * np.log(const.RBINS_ACC_EDGES[1:] / const.RBINS_ACC_EDGES[:-1]) / log_ratio,
    }


# ---------------------------------------------------------------------------
# Unfolding
# ---------------------------------------------------------------------------
def calc_spline(gr, xmin=None, xmax=None):
    """Smooth the MC->data efficiency correction with a log-linear spline."""
    det = gr.name[0]
    gr_filtered = gr.dropna()
    x = gr_filtered.index.get_level_values("rig").mid
    xmin = x.min() if const.X_LIMS_AVG2MC[det][0] is None else const.X_LIMS_AVG2MC[det][0]
    xmax = x.max() if const.X_LIMS_AVG2MC[det][1] is None else const.X_LIMS_AVG2MC[det][1]
    if len(x[(x > xmin) & (x < xmax)]) < 5:
        return None
    model = pyspl.fit_spline(
        x, gr_filtered["corr_avg2mc"], gr_filtered["corr_avg2mc_err"],
        mode="log-lin", extrapolation=["tangent", "constant"],
        lam=5, x_low=xmin, x_high=xmax,
    )
    gr["corr_avg2mc_spl"] = pyspl.eval_spline(model, gr.index.get_level_values("rig").mid)
    return gr


def unfold_iter(cr, eff, flux_model_x, flux_model_y, flux_model_y_err,
                rig_gen, mask, bins, acc_gen_t, weights_buf=None):
    """One unfolding iteration: reweight MC, recompute corrections and flux."""
    eff_mc, acc = proc.reweight(
        rig_gen, mask, bins, acc_gen_t,
        flux_model_x, flux_model_y, flux_model_y_err,
        weights_out=weights_buf,
    )

    eff["corr_avg2mc"] = eff["eff_avg"] / eff_mc["eff"]
    eff["corr_avg2mc_err"] = (
        (eff["eff_err_avg"] / eff["eff_avg"]) ** 2
        + (eff_mc["eff_err"] / eff_mc["eff"]) ** 2
    ) ** 0.5
    eff = eff.groupby(["det", "time_avg"], group_keys=False).apply(calc_spline)
    eff["corr"] = eff["corr_avg2mc_spl"] * eff["eff_daily2avg_spl"]

    fx = cr.copy()
    fx["corr_daily2avg"] = eff["eff_daily2avg_spl"].unstack("det").prod(axis=1)
    fx["corr_avg2mc"] = eff["corr_avg2mc_spl"].unstack("det").prod(axis=1)
    fx["corr"] = eff["corr"].unstack("det").prod(axis=1)
    fx["acc"] = acc_spline(fx.index.mid, acc.index.mid, acc["eff"].values)
    fx["fx"] = fx["cr"] / (fx["acc"] * fx["corr"])
    fx["fx_err"] = fx["cr_err"] / (fx["acc"] * fx["corr"])
    return fx


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
PROC_DATA: Optional[ProcessorData] = None
SHARED: dict = {}


def init_worker(shared_meta: dict) -> None:
    """ProcessPoolExecutor initializer: load ISS data and attach shared MC."""
    global PROC_DATA, SHARED

    PROC_DATA = ProcessorData(str(const.ISS_PATH), prefix=const.PREFIX)
    PROC_DATA.load()

    SHARED = attach_shared_arrays(shared_meta)
    SHARED["bins"] = {
        "in": SHARED["bins_in"],
        "tf": SHARED["bins_tf"],
        "l1": SHARED["bins_l1"],
        "tr": SHARED["bins_tr"],
        "acc": SHARED["bins_acc"],
    }
    # Per-worker reusable buffer for MC weights (NOT shared: each worker writes
    # its own values; sharing would race). Allocated once, reused across days
    # and unfold iterations.
    SHARED["weights_buf"] = np.empty(SHARED["rig_gen"].shape[0], dtype=np.float64)


def _flux_model_from(df: pd.DataFrame, y_col: str, y_err_col: str
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (x, y, y_err) numpy arrays from a binned flux/CR DataFrame."""
    return (
        df.index.mid.to_numpy(copy=True),
        df[y_col].to_numpy(copy=True),
        df[y_err_col].to_numpy(copy=True),
    )


def process_day(t: pd.Timestamp):
    """Run the iterative unfolding for a single day `t`."""
    # print(f"Processing {t}")
    try:
        cr = PROC_DATA.count_rate.loc[t, ["cr", "cr_err"]]
        eff = PROC_DATA.effs_daily.xs(t, level="time")
    except KeyError:
        print(f"No data for {t}")
        return None

    cr = cr.dropna()
    cr = cr[cr.index.mid < RIG_MAX_USED]

    rig_gen = SHARED["rig_gen"]
    mask = SHARED["mask"]
    bins = SHARED["bins"]
    acc_gen_t = SHARED["acc_gen_t"]
    weights_buf = SHARED["weights_buf"]

    fmx, fmy, fmy_err = _flux_model_from(cr, "cr", "cr_err")
    fx = unfold_iter(cr, eff, fmx, fmy, fmy_err, rig_gen, mask, bins, acc_gen_t, weights_buf)

    for _ in range(1, MAX_ITERATIONS):
        fmx, fmy, fmy_err = _flux_model_from(fx, "fx", "fx_err")
        fx_new = unfold_iter(cr, eff, fmx, fmy, fmy_err, rig_gen, mask, bins, acc_gen_t, weights_buf)
        converged = np.all(np.abs(fx_new["fx"] / fx["fx"] - 1) < CONVERGENCE_TOL)
        fx = fx_new
        if converged:
            break
    return fx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str]) -> None:
    if (argv[1] == 'recreate'):
        print("Recreating efficiencies...")
        proc_data = ProcessorData(str(const.ISS_PATH), prefix=const.PREFIX)
        proc_data.init_count_rate()
        proc_data.load_efficiencies()
        proc_data.calc_efficiencies()
        proc_data.save()
        print("Efficiencies recreated successfully")
        return

    threading.Thread(target=monitor_memory, daemon=True).start()

    shared_objects = load_mc(str(const.MC_PATH))

    shms: list[shared_memory.SharedMemory] = []
    shared_meta: dict = {}
    for key, arr in shared_objects.items():
        shm, meta = make_shared_array(arr)
        shms.append(shm)
        shared_meta[key] = meta

    del shared_objects
    gc.collect()

    t1 = pd.Timestamp(argv[1], tz="UTC")
    t2 = pd.Timestamp(argv[2], tz="UTC")
    tt = pd.date_range(t1, t2, freq="D")
    tt = np.random.choice(tt, size=1000, replace=False)
    cr = ProcessorData(str(const.ISS_PATH), prefix=const.PREFIX)
    cr.load()
    cr = cr.count_rate.index.get_level_values('time').unique()
    tt = tt[np.isin(tt, cr)]

    results: list = {}
    executor: Optional[ProcessPoolExecutor] = None
    futures: dict = {}
    try:
        executor = ProcessPoolExecutor(
            max_workers=NUM_WORKERS,
            initializer=init_worker,
            initargs=(shared_meta,),
        )
        futures = {executor.submit(process_day, t): t for t in tt}
        for future in tqdm(as_completed(futures), total=len(futures)):
            t = futures[future]
            try:
                results[t] = future.result()
            except Exception as e:
                print(f"Error for {t}: {e}")
    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping workers...")
        for f in futures:
            f.cancel()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        out_path = const.DATA_DIR / f"flux_daily_{argv[1]}_{argv[2]}.pkl"
        results = pd.concat(results, names=["time"])
        pd.to_pickle(results, out_path)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        for shm in shms:
            shm.close()
            shm.unlink()


if __name__ == "__main__":
    main(sys.argv)
