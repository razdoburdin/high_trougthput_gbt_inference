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

configs = {
    "a9a": {
        "objective": "binary:logistic",
        "max_depth": 8,
        "n_estimators": 200
    },
    "airline": {
        "objective": "binary:logistic",
        "max_depth": 8,
        "n_estimators": 100
    },
    "airline-ohe": {
        "objective": "binary:logistic",
        "max_depth": 8,
        "n_estimators": 1000
    },
    "creditcard": {
        "objective": "binary:logistic",
        "max_depth": 10,
        "n_estimators": 100
    },
    "california_housing": {
        "objective": "reg:squarederror",
        "max_depth": 5,
        "n_estimators": 100
    },
    "higgs": {
        "objective": "binary:logistic",
        "max_depth": 8,
        "n_estimators": 1000
    },
    "letters": {
        "objective": "multi:softprob",
        "max_depth": 6,
        "n_estimators": 1000
    },
    "msrank": {
        "objective": "multi:softprob",
        "max_depth": 6,
        "n_estimators": 20
    },
    "year_prediction_msd": {
        "objective": "reg:squarederror",
        "max_depth": 8,
        "n_estimators": 20
    },
    "susy": {
        "objective": "binary:logistic",
        "max_depth": 8,
        "n_estimators": 1000
    },
    "epsilon": {
        "objective": "binary:logistic",
        "max_depth": 8,
        "n_estimators": 1000
    },
    "covtype_binary": {
        "objective": "binary:logistic",
        "max_depth": 8,
        "n_estimators": 1000
    },
    "hepmass": {
        "objective": "binary:logistic",
        "max_depth": 8,
        "n_estimators": 1000
    },
    "covtype_full": {
        "objective": "binary:logistic",
        "max_depth": 8,
        "n_estimators": 1000
    },
    "kddcup99": {
        "objective": "binary:logistic",
        "max_depth": 8,
        "n_estimators": 1000
    },
    "miniboone": {
        "objective": "binary:logistic",
        "max_depth": 8,
        "n_estimators": 1000
    },
    "poker": {
        "objective": "binary:logistic",
        "max_depth": 8,
        "n_estimators": 1000
    }
}