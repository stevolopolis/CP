#! /bin/bash

python app/main.py \
    --net NGP \
    --n-levels 6 \
    --log2-n-features 22 \
    --feature-dim 2 \
    --base-resolution 16 \
    --finest-resolution 256000 \
    --hidden-features 64 \
    --hidden-layers 2 \
    --out-features 1 \
    --num-samples 100000 \
    --batch-size 512 \
    --lr 0.00005 \
    --device cuda:0 \
    --exp-name ngp_hash_visualizer_v1_1 \
    --dataset-path data/armadillo.obj \
    --matcap-path data/matcap/rainbow.png

python app/main.py \
    --net OctreeSDF \
    --num-lods 5 \
    --dataset-path data/armadillo.obj \
    --epochs 250 \
    --exp-name armadillo \
    --matcap-path data/matcap/rainbow.png

python app/main.py \
    --net SIREN \
    --num-layers 5\
    --dim-hidden 256\
    --w0 30.0\
    --w0-initial 30.0\
    --lr 0.00001 \
    --dataset-path data/armadillo.obj \
    --epochs 250 \
    --exp-name siren \
    --matcap-path data/matcap/rainbow.png

python app/main.py \
    --net TACO \
    --num-layers 10 \
    --dim-in 3 \
    --dim-out 1 \
    --dim-hidden 512 \
    --w0 30.0 \
    --w0-initial 30.0 \
    --lr 0.00001 \
    --data-resolution 4096 \
    --batch-size 4096 \
    --num-samples 250000 \
    --dataset-path data/armadillo.obj \
    --epochs 500 \
    --device cuda:1 \
    --exp-name taco_4096res_10layers_512hidden_bigBatch_altCoord \
    --matcap-path data/matcap/rainbow.png