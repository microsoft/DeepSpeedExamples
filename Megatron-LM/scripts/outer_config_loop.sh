#!/bin/bash


#for i in {2..7}; do 
#    ls scripts/config$i.dat
#    echo scripts/ds_gpt_experiment_loop.sh scripts/config$i.dat
#    bash scripts/ds_gpt_experiment_loop.sh scripts/config$i.dat
#done
#
#echo "scripts/exp_170b.dat"
#bash scripts/ds_gpt_experiment_loop.sh scripts/exp_170b.dat

for config in `ls --reverse scripts/hyperscale/*.dat`; do
    echo scripts/ds_gpt_experiment_loop.sh $config
    bash scripts/ds_gpt_experiment_loop.sh $config
done

#bash scripts/ds_gpt_experiment_loop.sh scripts/config6sweep/1p5Bmp8.dat
#bash next.sh

