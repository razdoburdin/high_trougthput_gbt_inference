#!/usr/bin/env bash

for fw in xgboost daal4py tl2cgen treelite onnx; do
    /usr/bin/time \
        numactl -N 0 -m 0 \
            python run_inference.py --framework=$fw &> $fw.log
done
