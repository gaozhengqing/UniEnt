#!/bin/bash

set -Eeuxo pipefail

while getopts ":g:" opt; do
    case "$opt" in
        g)
            gpu="$OPTARG"
            ;;
    esac
done

for adaptation in source norm; do
    CUDA_VISIBLE_DEVICES=$gpu python main.py --adaptation $adaptation --save_dir "./output"
done

CUDA_VISIBLE_DEVICES=$gpu python main.py --adaptation cotta --lr 0.01 --save_dir "./output"

# Tent, EATA, OSTTA
for adaptation in tent eata ostta; do
    for alpha in 1.0 0.5 0.2 0.1; do
            CUDA_VISIBLE_DEVICES=$gpu python main.py --adaptation $adaptation --save_dir "./output" --alpha $alpha
    done
done

# Tent, EATA, OSTTA + UniEnt, UniEnt+
for adaptation in tent eata ostta; do
    for alpha1 in 1.0 0.5 0.2 0.1; do
        for alpha2 in 1.0 0.5 0.2 0.1; do
            for criterion in ent_ind_ood ent_unf; do
                CUDA_VISIBLE_DEVICES=$gpu python main.py --adaptation $adaptation --save_dir "./output" --alpha $alpha1 $alpha2 --criterion $criterion
            done
        done
    done
done
