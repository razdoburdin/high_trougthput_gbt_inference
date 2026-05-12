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

import os
import treelite
import tl2cgen

import datasets
import models

if __name__ == '__main__':
    for name in datasets.dataset_loaders:
        print(f"Converting: {name}")

        booster, objective = models.get_model(name)
        treelite_model = treelite.frontend.from_xgboost(booster)
        libpath = models.model_path("treelite", name, ext=".so")
        tl2cgen.export_lib(treelite_model, toolchain="gcc", libpath=libpath,
                           params={"parallel_comp": os.cpu_count()})
