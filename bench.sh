#!/usr/bin/env bash

for fw in xgboost daal4py treelite onnx xgboost_sycl; do
    /usr/bin/time python run_inference.py --framework=$fw &> $fw.log
done
