import os
import json

import os.path
from pathlib import Path
from moabb.paradigms import LeftRightImagery
from moabb.datasets import (
    BNCI2014001,
    Cho2017,
    Lee2019_MI,
    MunichMI,
    PhysionetMI,
    Shin2017A,
    Schirrmeister2017,
    Weibo2014,
    Zhou2016,
)

P_VAL = 0.05
N_FEATURES = 10
LATERALIZATION_METRIC = ["local_laterality", "segregation", "integration"]
EXCLUDE_CHANNELS = ["F9", "F10", "FT9", "FT10", "M1", "M2"]
DATASETS = [
    BNCI2014001(),
    Cho2017(),
    Lee2019_MI(),
    # MunichMI(),
    PhysionetMI(),
    Shin2017A(accept=True),
    Schirrmeister2017(),
    Weibo2014(),
    Zhou2016(),
]  # for rh vs lh


class ConfigPath:
    RES_DIR = Path(os.getcwd(), "../results")
    RES_DIR.mkdir(parents=True, exist_ok=True)

    RES_CLASSIFY_DIR = RES_DIR / "classification"
    RES_CLASSIFY_DIR.mkdir(parents=True, exist_ok=True)

    RES_CSP = RES_DIR / "csp_features"
    RES_CSP.mkdir(parents=True, exist_ok=True)

    RES_FC = RES_DIR / "functionalconnectivity"
    RES_FC.mkdir(parents=True, exist_ok=True)

    RES_NET = RES_DIR / "netmetric"
    RES_NET.mkdir(parents=True, exist_ok=True)

    RES_3DPLOT = RES_DIR / "3Dplot"
    RES_3DPLOT.mkdir(parents=True, exist_ok=True)


# Application settings
def load_config(jsonfile="params.json"):
    if jsonfile in os.listdir(Path(os.getcwd())):
        with open(Path(os.getcwd()) / jsonfile) as jsonfile:
            params = json.load(jsonfile)
            paradigm = LeftRightImagery(
                fmin=params["fmin"], fmax=params["fmax"]
            )  # for rh vs lh
            return params, paradigm
    else:
        # Handle the case where the JSON file is not found
        print(f"Error: {jsonfile} not found.")
        return None
