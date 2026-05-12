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

from pathlib import Path
import xgboost as xgb
import json

def model_path(framework, name, ext=".json"):
    model_folder = "models"
    return f"{model_folder}/{framework}_{name}_model{ext}"

def get_model(name):
    path2model = model_path("xgboost", name)
    path = Path(path2model)
    if path.exists():
        booster = xgb.Booster()
        booster.load_model(path2model)

        config = json.loads(booster.save_config())
        return booster, config["learner"]["objective"]["name"]
    else:
        raise ValueError("Model file doesn't exist")
