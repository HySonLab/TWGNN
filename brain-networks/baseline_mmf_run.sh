data=adj_mx_brain_sub5
adj=data/sensor_graph/${data}.pkl
N=116
how='baseline'
dir=data/ECG_sub5/wavelets/${how}

for L in {20..50..10}
do
    dim=$(( N - L ))
    name=${data}.${how}.L.${L}.dim.${dim}
    echo $name
    python -W ignore baseline_mmf_run.py --L=${L} --dim=${dim} --name=${name} --adj=${adj} --dim=${dim} --dir=${dir}
done
