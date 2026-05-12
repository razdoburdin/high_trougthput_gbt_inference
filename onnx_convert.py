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
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

import datasets
import models

if __name__ == '__main__':
    for name in datasets.dataset_loaders:
        print(f"Converting: {name}")

        booster, objective = models.get_model(name)
        [X, _] = datasets.get_data(name, "test")
        n_features = X.shape[1]

        initial_type = [("input", FloatTensorType([None, n_features]))]
        onnx_model = onnxmltools.convert_xgboost(booster, initial_types=initial_type)

        onnx_path = models.model_path("onnx", name, ext=".onnx")
        onnxmltools.utils.save_model(onnx_model, onnx_path)
