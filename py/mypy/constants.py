import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
INCLUDE_DIR = PROJECT_PATH / "include"
BUILD_DIR = PROJECT_PATH / "build"
DATA_DIR = PROJECT_PATH / "data"
ISS_PATH = DATA_DIR / "data.root"
MC_PATH = DATA_DIR / "mc_05100_f8.root"

OUTPUT_DIR = PROJECT_PATH / "output_new"

RIG_MIN = 0.5
RIG_MAX = 100.0
RBINS_EDGES = np.array([
    0.6,
    0.8, 
    1.00, 
    1.16, 
    1.33, 
    1.51, 
    1.71, 
    1.92, 
    2.15, 
    2.40, 
    2.67, 
    2.97, 
    3.29,
    3.64, 
    4.02, 
    4.43, 
    4.88, 
    5.37, 
    5.90, 
    6.47, 
    7.09, 
    7.76, 
    8.48, 
    9.26,
    10.1, 
    11.0, 
    12.0, 
    13.0, 
    14.1, 
    15.3, 
    16.6, 
    18.0, 
    19.5, 
    21.1, 
    22.8,
    24.7, 
    26.7, 
    28.8, 
    31.1, 
    33.5, 
    36.1, 
    38.9, 
    41.9, 
    45.1, 
    48.5, 
    52.2,
    56.1, 
    60.3, 
    64.8, 
    69.7, 
    74.9, 
    80.5, 
    86.5, 
    93.0, 
    100,  
    108,  
    116,
    125,  
    135,  
    147,  
    160,  
    175,  
    192,  
    211,  
    233,  
    259,  
    291,  
    330,
    379,  
    441,  
    525,  
    643, 
    822, 
    1130, 
    1800, 
    3000
])
# RBINS_ACC_EDGES = np.r_[np.logspace(np.log10(RIG_MIN), np.log10(1.0), 500), np.logspace(np.log10(1.0), np.log10(RIG_MAX), 1500)[1:]]
RBINS_ACC_EDGES = np.logspace(np.log10(RIG_MIN), np.log10(RIG_MAX), 2000)
RBINS = pd.IntervalIndex.from_breaks(RBINS_EDGES, name="rig", closed="left")
RBINS_ACC = pd.IntervalIndex.from_breaks(RBINS_ACC_EDGES, name="rig_acc", closed="left")

TMIN_GLOB = pd.Timestamp("2011-05-23", tz="UTC") # start
TMAX_GLOB = pd.Timestamp("2025-10-01", tz="UTC") # end

# IN_AVG_PERIODS = pd.DatetimeIndex([
#     TMIN_GLOB,
#     pd.Timestamp("2011-07-21", tz="UTC"), # Calibration period
#     pd.Timestamp("2011-12-01", tz="UTC"), # 6 ladder on X dead
#     pd.Timestamp("2014-05-08", tz="UTC"), # one ladder on Y dead
#     pd.Timestamp("2017-03-01", tz="UTC"), # one ladder on Y dead
#     pd.Timestamp("2019-09-24", tz="UTC"), # ISS power cut
#     TMAX_GLOB,
# ])

TR_AVG_PERIODS = pd.DatetimeIndex([
    TMIN_GLOB,
    pd.Timestamp("2013-11-26", tz="UTC"), # TOF-PMT config
    pd.Timestamp("2016-02-26", tz="UTC"), # TOF-PMT config
    pd.Timestamp("2020-02-18", tz="UTC"), # TOF-PMT config
    pd.Timestamp("2021-05-03", tz="UTC"), # TOF-PMT config
    pd.Timestamp("2021-11-02", tz="UTC"), # TOF-PMT config
    pd.Timestamp("2023-02-02", tz="UTC"), # TOF-PMT config
    TMAX_GLOB,
])

IN_AVG_PERIODS = pd.DatetimeIndex([
    TMIN_GLOB,
    pd.Timestamp("2011-07-21", tz="UTC"), # Calibration period
    pd.Timestamp("2011-12-01", tz="UTC"), # 6 ladder on X dead
    pd.Timestamp("2012-10-29", tz="UTC"), # 6 ladder on X dead
    pd.Timestamp("2014-05-08", tz="UTC"), # one ladder on Y dead
    pd.Timestamp("2017-02-28", tz="UTC"), # one ladder on Y dead
    # pd.Timestamp("2018-02-27", tz="UTC"), # one ladder on Y dead
    pd.Timestamp("2018-12-04", tz="UTC"), # Something
    pd.Timestamp("2019-05-03", tz="UTC"), # Something
    pd.Timestamp("2019-09-24", tz="UTC"), # Something
    pd.Timestamp("2020-01-27", tz="UTC"), # Something
    pd.Timestamp("2021-03-05", tz="UTC"), # Something
    TMAX_GLOB,
])

L1_AVG_PERIODS = pd.DatetimeIndex([
    TMIN_GLOB,
    pd.Timestamp("2011-07-21", tz="UTC"), # Calibration period
    pd.Timestamp("2011-12-01", tz="UTC"), # 6 ladder on X dead
    pd.Timestamp("2012-10-29", tz="UTC"), # 6 ladder on X dead
    pd.Timestamp("2014-05-08", tz="UTC"), # one ladder on Y dead
    pd.Timestamp("2017-02-28", tz="UTC"), # one ladder on Y dead
    # pd.Timestamp("2018-02-27", tz="UTC"), # one ladder on Y dead
    pd.Timestamp("2018-12-04", tz="UTC"), # Something
    pd.Timestamp("2019-05-03", tz="UTC"), # Something
    pd.Timestamp("2019-09-24", tz="UTC"), # Something
    pd.Timestamp("2020-01-27", tz="UTC"), # Something
    pd.Timestamp("2021-03-05", tz="UTC"), # Something
    TMAX_GLOB,
])

TF_AVG_PERIODS = pd.DatetimeIndex([
    TMIN_GLOB,
    TMAX_GLOB,
])

AVG_PERIODS = {
    'in': IN_AVG_PERIODS,
    'tr': TR_AVG_PERIODS,
    'l1': L1_AVG_PERIODS,
    'tf': TF_AVG_PERIODS,
}

X_LIMS_DAY2AVG = {
    'in': (None, 5.0),
    'l1': (1.0, 10.0),
    'tf': (None, 10.0),
    'tr': (None, 10.0),
}

X_LIMS_AVG2MC = {
    'in': (None, 5.0),
    'l1': (1.0, 60.0),
    'tf': (None, None),
    'tr': (None, None),
}


IN = 0
TF = 1
L1 = 2
TR = 3
DETS = {
    'in': IN,
    'tf': TF,
    'l1': L1,
    'tr': TR,
}

IN_P = 1 << 0
IN_T = 1 << 1
TF_P = 1 << 2
TF_T = 1 << 3
L1_P = 1 << 4
L1_T = 1 << 5
MAIN_PHYS = 1 << 6
MAIN_UNB = 1 << 7

EFF_IDX = pd.MultiIndex.from_product([DETS.keys(), RBINS], names=['det', 'rig'])