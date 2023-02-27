data=adj_mx_brain_sub5
adj=data/sensor_graph/${data}.pkl
N=116
learning_rate=5e-1
epochs=3000
how='learnable'
dir=data/ECG_sub5/wavelets/${how}

drop=1
K=2
for L in {20..50..10}
do
    dim=$(( N - L * drop ))
    name=${data}.${how}.L.${L}.K.${K}.drop.${drop}.dim.${dim}
    echo $name
    python -W ignore learnable_mmf_train.py --L=${L} --K=${K} --dim=${dim} --name=${name} --adj=${adj} --learning_rate=$learning_rate --epochs=${epochs} --drop=${drop} --dir=${dir}
done
