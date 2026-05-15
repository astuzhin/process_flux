import htcondor
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import shutil
import os
import re
import subprocess

def extract_start_ts(path):
    nums = re.findall(r'\d+', path.name)
    return int(nums[0]), int(nums[1]), int(nums[2])

proj_path = Path("/lustre02/user/astuzhin/science/projects/ProtonSelection/FluxAnalysis/process_flux")

log_path = proj_path / "logs"
shutil.rmtree(log_path, ignore_errors=True)
log_path.mkdir(parents=True, exist_ok=True)

# data_path = proj_path / "data"
# shutil.rmtree(data_path / "flux*.pkl", ignore_errors=True)

t1 = pd.Timestamp("2012-01-01", tz="UTC")
t2 = pd.Timestamp("2025-10-01", tz="UTC")
times = pd.date_range(start=t1, end=t2, freq=pd.Timedelta(days=1))[::500]

itemdata = []
for i in range(len(times) - 1):
    t1 = times[i].strftime("%Y-%m-%d")
    t2 = times[i+1].strftime("%Y-%m-%d")
    itemdata.append({
        "t1": t1,
        "t2": t2,
    })

job = htcondor.Submit({
    "executable": f"{proj_path.resolve().as_posix()}/wrapper.sh",
    "arguments": "$(t1) $(t2)",
    "output": f"{log_path.resolve().as_posix()}/condor_$(Process).out",
    "error": f"{log_path.resolve().as_posix()}/condor_$(Process).err",
    "log": f"{log_path.resolve().as_posix()}/condor.log",
    "request_cpus": "32"
})

schedd = htcondor.Schedd()
submit_result = schedd.submit(job, itemdata=iter(itemdata))
print(submit_result.cluster())