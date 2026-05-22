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
import time
import argparse
import daal4py
import tl2cgen
import treelite
import treelite.gtil
import onnxruntime as ort

import datasets
import models
import results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=str, default="xgboost")
    args = parser.parse_args()

    result = {}
    try:
        for name in datasets.dataset_loaders:
            [X, y] = datasets.get_data(name, "test")
            n_samples = X.shape[0]

            booster, objective = models.get_model(name)

            if args.framework == "xgboost":
                import xgboost as xgb
                begin = time.perf_counter()
                out = booster.predict(xgb.DMatrix(X))
                end = time.perf_counter()

            elif args.framework == "xgboost_sycl":
                import xgboost as xgb
                booster.set_param({"device": "sycl:gpu"})
                begin = time.perf_counter()
                out = booster.predict(xgb.DMatrix(X))
                end = time.perf_counter()

            elif args.framework == "daal4py":
                X = np.asarray(X, np.float32)
                model_daal = daal4py.mb.convert_model(booster)
                begin = time.perf_counter()
                out = model_daal.predict(X)
                end = time.perf_counter()

            elif args.framework == "tl2cgen":
                X = np.asarray(X, np.float32)
                libpath = models.model_path("tl2cgen", name, ext=".so")
                predictor = tl2cgen.Predictor(libpath)
                begin = time.perf_counter()
                out = predictor.predict(tl2cgen.DMatrix(X))
                end = time.perf_counter()

            elif args.framework == "treelite":
                X = np.asarray(X, np.float32)
                tl_path = models.model_path("treelite", name, ext=".tmd")
                tl_model = treelite.Model.deserialize(tl_path)
                begin = time.perf_counter()
                out = treelite.gtil.predict(tl_model, X, nthread=-1)
                end = time.perf_counter()

            elif args.framework == "onnx":
                X = np.asarray(X, np.float32)
                onnx_path = models.model_path("onnx", name, ext=".onnx")
                session = ort.InferenceSession(onnx_path)
                input_name = session.get_inputs()[0].name
                begin = time.perf_counter()
                out = session.run(None, {input_name: X})
                end = time.perf_counter()

            else:
                raise ValueError("Unknown framework")

            time_sec = end - begin
            throughput = n_samples / time_sec
            result[name] = {
                "n_samples": n_samples,
                "time_sec": round(time_sec, 6),
                "throughput [samples/sec]": round(throughput)
            }
    except ValueError as e:
        print(e)

    results.save_result(result, args.framework)
