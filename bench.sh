#!/usr/bin/env bash

for fw in xgboost daal4py treelite onnx xgboost_sycl; do
# for fw in treelite onnx; do
    python run_inference.py --framework=$fw
done
