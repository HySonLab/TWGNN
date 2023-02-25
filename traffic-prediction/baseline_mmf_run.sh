data=adj_mx_bay
adj=data/sensor_graph/${data}.pkl
N=325
how='baseline'

for L in {50..200..10}
do
    dim=$(( N - L ))
    name=${data}.${how}.L.${L}.dim.${dim}
    echo $name
    python -W ignore baseline_mmf_run.py --L=${L} --dim=${dim} --name=${name} --adj=${adj} --dim=${dim}
done
