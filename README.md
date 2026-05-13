Steps to reproduce results:

## Install miniforge
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
chmod +x Miniforge3-Linux-x86_64.sh
./Miniforge3-Linux-x86_64.sh -b
eval "$(miniforge3/bin/conda shell.bash hook)"
```

## Clone repo
```
git clone https://github.com/razdoburdin/high_trougthput_gbt_inference.git
```

## Create env
```
cd high_trougthput_gbt_inference
conda env create --name gbt -f environment.yml -y
conda activate gbt
```

## Convert xgboost model to treelite and onnx
```
python treelite_convert.py
python onnx_convert.py
```
