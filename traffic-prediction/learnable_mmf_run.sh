data=adj_mx
adj=data/sensor_graph/${data}.pkl
N=207
learning_rate=5e-1
epochs=3000
how='learnable'

drop=3
K=4
for L in {20..100..10}
do
    dim=$(( N - L * drop ))
    name=${data}.${how}.L.${L}.K.${K}.drop.${drop}.dim.${dim}
    echo $name
    python -W ignore learnable_mmf_train.py --L=${L} --K=${K} --dim=${dim} --name=${name} --adj=${adj} --learning_rate=$learning_rate --epochs=${epochs} --drop=${drop}
done
