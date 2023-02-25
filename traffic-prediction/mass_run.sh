python -c 'import yaml;f=open("data/model/dcrnn_bay.yaml");y=yaml.safe_load(f);y["model"]["wavelet_name"] = "data/PEMS-BAY/wavelets/baseline/adj_mx_bay.baseline.L.50.dim.275"; fw=open("data/model/dcrnn_bay.yaml","w");yaml.dump(y,fw, default_flow_style=False, sort_keys=False)'
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_bay.yaml 

python -c 'import yaml;f=open("data/model/dcrnn_bay.yaml");y=yaml.safe_load(f);y["model"]["wavelet_name"] = "data/PEMS-BAY/wavelets/baseline/adj_mx_bay.baseline.L.60.dim.265"; fw=open("data/model/dcrnn_bay.yaml","w");yaml.dump(y,fw, default_flow_style=False, sort_keys=False)'
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_bay.yaml 

python -c 'import yaml;f=open("data/model/dcrnn_bay.yaml");y=yaml.safe_load(f);y["model"]["wavelet_name"] = "data/PEMS-BAY/wavelets/baseline/adj_mx_bay.baseline.L.70.dim.255"; fw=open("data/model/dcrnn_bay.yaml","w");yaml.dump(y,fw, default_flow_style=False, sort_keys=False)'
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_bay.yaml 

python -c 'import yaml;f=open("data/model/dcrnn_bay.yaml");y=yaml.safe_load(f);y["model"]["wavelet_name"] = "data/PEMS-BAY/wavelets/baseline/adj_mx_bay.baseline.L.80.dim.245"; fw=open("data/model/dcrnn_bay.yaml","w");yaml.dump(y,fw, default_flow_style=False, sort_keys=False)'
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_bay.yaml 

python -c 'import yaml;f=open("data/model/dcrnn_bay.yaml");y=yaml.safe_load(f);y["model"]["wavelet_name"] = "data/PEMS-BAY/wavelets/baseline/adj_mx_bay.baseline.L.90.dim.235"; fw=open("data/model/dcrnn_bay.yaml","w");yaml.dump(y,fw, default_flow_style=False, sort_keys=False)'
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_bay.yaml 

python -c 'import yaml;f=open("data/model/dcrnn_bay.yaml");y=yaml.safe_load(f);y["model"]["wavelet_name"] = "data/PEMS-BAY/wavelets/baseline/adj_mx_bay.baseline.L.100.dim.225"; fw=open("data/model/dcrnn_bay.yaml","w");yaml.dump(y,fw, default_flow_style=False, sort_keys=False)'
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_bay.yaml 