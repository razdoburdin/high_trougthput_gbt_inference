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
import xgboost as xgb
from pathlib import Path

import datasets
import models

if __name__ == '__main__':
    for name in datasets.dataset_loaders:
        print(f"Training: {name}")

        [X, y] = datasets.get_data(name, "train")
        dmatrix = xgb.DMatrix(X, y)

        params = models.configs[name]
        n_estimators = params.pop("n_estimators")
        if "multi:" in params["objective"]:
            num_class = len(np.unique(y))
            params.update({"num_class": num_class})

        booster = xgb.train(params,
                            xgb.DMatrix(X, y),
                            num_boost_round = n_estimators)

        booster.save_model(models.model_path("xgboost", name))