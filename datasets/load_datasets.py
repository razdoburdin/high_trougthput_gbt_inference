# ===============================================================================
# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import numpy as np
from pathlib import Path
from typing import Callable, Dict

from .loader_classification import (a_nine_a, airline, airline_ohe, creditcard, higgs,
                                    covtype_binary, susy, epsilon,
                                    hepmass, covtype_full, kddcup99, miniboone, poker)
from .loader_multiclass import (letters, msrank)
from .loader_regression import (california_housing, year_prediction_msd)

dataset_loaders: Dict[str, Callable[[Path], bool]] = {
    "a9a": a_nine_a,
    "airline": airline,
    "airline-ohe": airline_ohe,
    "california_housing": california_housing,
    "creditcard": creditcard,
    "higgs": higgs,
    "susy": susy,
    "covtype_binary": covtype_binary,
    "hepmass": hepmass,
    "covtype_full": covtype_full,
    "kddcup99": kddcup99,
    "miniboone": miniboone,
    "poker": poker,
    "msrank": msrank,
    "year_prediction_msd": year_prediction_msd
}

def is_loaded(folder: Path, name: str, stage: str):
    path_x = Path(f"{folder}/{name}_x_{stage}.npy")
    path_y = Path(f"{folder}/{name}_y_{stage}.npy")

    return path_x.exists() and path_y.exists()

def get_data(name: str, stage: str):
    data_folder = "data"
    loader = dataset_loaders[name]

    if (not is_loaded(data_folder, name, stage)):
        loader(data_folder)

    X = np.load(f"{data_folder}/{name}_x_{stage}.npy", allow_pickle=True)
    y = np.load(f"{data_folder}/{name}_y_{stage}.npy", allow_pickle=True)

    return [X, y]