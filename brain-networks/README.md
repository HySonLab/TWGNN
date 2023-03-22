# Fast Temporal Wavelet Graph Neural Networks

This is a PyTorch implementation of Fast Temporal Wavelet Graph Neural Networks in the following paper: \
Nguyen, D. T., Nguyen, M. D. T., Hy, T. S., & Kondor, R. (2023). [Fast Temporal Wavelet Graph Neural Networks](https://arxiv.org/abs/2302.08643). arXiv preprint arXiv:2302.08643.
## Requirements
* torch
* scipy>=0.19.0
* numpy>=1.12.1
* pandas>=0.19.2
* pyyaml
* statsmodels
* torch
* tables
* future

Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Dataset
[*Nature* article](https://www.nature.com/articles/s41597-022-01280-y)

Download from [DANDI archive](https://dandiarchive.org/dandiset/000055/0.220127.0436)


## Data preparation
Here is an article about [Using HDF5 with Python](https://medium.com/@jerilkuriakose/using-hdf5-with-python-6c5242d08773).

Run the following commands to generate train/test/val dataset at  `data/ECG_sub5/{train,val,test}.npz`.
```bash
# Create data directories
mkdir -p data/ECG_sub5

# bash script generate data
sh generate_training_data.sh
```

## Graph Construction
 An example with a network whose distances are defined from the xyz coordinates of the electrodes (see `data/sensor_graph/distance_ecg_sub5.csv`).
```bash
python -m scripts.gen_adj_mx  --sensor_ids_filename=data/sensor_graph/electrodes_id_sub5.txt --normalized_k=0.1\
    --output_pkl_filename=data/sensor_graph/adj_mx_brain_sub5.pkl
```

## Generate wavelet bases
Following the code in this [MMF algorithm](https://github.com/risilab/Learnable_MMF) (paper attached).
For example, run baseline MMF algorithm by
```bash
sh baseline_mmf_run.sh
```
Mother/father wavelets are saved into the folder `data/ECG_sub5/wavelets/baseline/`

## Model Training
```bash
sh twnn_run.sh
```

## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:
```
@article{nguyen2023fast,
  title={Fast Temporal Wavelet Graph Neural Networks},
  author={Nguyen, Duc Thien and Nguyen, Manh Duc Tuan and Hy, Truong Son and Kondor, Risi},
  journal={arXiv preprint arXiv:2302.08643},
  year={2023}
}
```
