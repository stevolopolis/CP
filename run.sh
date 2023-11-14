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
    --dataset-path ../../datasets/ShapeNetCore.v2/02691156/6f473d567942897b9908db9f2ff495fe/models/model_normalized.obj \
    --epochs 250 \
    --data-resolution 1024 \
    --batch-size 4096 \
    --device cuda:1 \
    --exp-name nglod_shapenet \
    --matcap-path data/matcap/rainbow.png \
    --num-samples 250000

python app/main.py \
    --net SIREN \
    --num-layers 10 \
    --dim-hidden 512\
    --w0 30.0\
    --w0-initial 30.0\
    --lr 0.00001 \
    --dataset-path ../../datasets/ShapeNetCore.v2/02691156/6f473d567942897b9908db9f2ff495fe/models/model_normalized.obj \
    --epochs 250 \
    --data-resolution 1024 \
    --batch-size 4096 \
    --num-samples 250000 \
    --dataset-path ../../datasets/ShapeNetCore.v2/03928116/c7ab660e1fde9bb48ce2930380f4c6e7/models/model_normalized.obj \
    --epochs 500 \
    --device cuda:1 \
    --exp-name siren_shapenet03928116_256h_v2 \
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
    --data-resolution 1024 \
    --batch-size 4096 \
    --num-samples 250000 \
    --dataset-path ../../datasets/ShapeNetCore.v2/03928116/c7ab660e1fde9bb48ce2930380f4c6e7/models/model_normalized.obj \
    --epochs 500 \
    --device cuda:2 \
    --exp-name taco_shapenet03928116_256h_v2 \
    --matcap-path data/matcap/rainbow.png

/datasets/ShapeNetCore.v2/02691156/6f473d567942897b9908db9f2ff495fe/models/model_normalized.obj
/datasets/ShapeNetCore.v2/03928116/c7ab660e1fde9bb48ce2930380f4c6e7/models/model_normalized.obj

python app/main.py \
    --net SIREN \
    --num-layers 10 \
    --dim-hidden 512\
    --w0 30.0\
    --w0-initial 30.0\
    --lr 0.00001 \
    --data-resolution 1024 \
    --batch-size 100000 \
    --num-samples 1000000 \
    --min-dis 0.0001 \
    --dataset-path ../../datasets/samples \
    --raw-obj-path Cone.npz \
    --mesh-dataset AltDataset \
    --epochs 500 \
    --device cuda:1 \
    --exp-name siren_cone_512h \
    --matcap-path data/matcap/rainbow.png \
    --use-wandb 1 \
    --wandb-group towaki

python app/main.py \
    --net TACO \
    --num-layers 10 \
    --dim-in 3 \
    --dim-out 1 \
    --dim-hidden 512 \
    --w0 30.0 \
    --w0-initial 30.0 \
    --lr 0.00005 \
    --data-resolution 1024 \
    --batch-size 100000 \
    --num-samples 1000000 \
    --dataset-path ../../datasets/samples \
    --raw-obj-path Cone.npz \
    --mesh-dataset AltDataset \
    --epochs 500 \
    --device cuda:1 \
    --exp-name taco_cone_512h_withTV \
    --matcap-path data/matcap/rainbow.png \
    --use-wandb 1 \
    --wandb-group towaki

python app/sdf_renderer.py \
    --net SIREN \
    --num-layers 10 \
    --dim-in 3 \
    --dim-out 1 \
    --dim-hidden 512 \
    --w0 30.0 \
    --w0-initial 30.0 \
    --lr 0.00005 \
    --data-resolution 1024 \
    --min-dis 0.0000001 \
    --batch-size 100000 \
    --num-samples 1000000 \
    --pretrained _results/models/siren_spike_512h.pth \
    --dataset-path ../../datasets/samples \
    --raw-obj-path Spike.npz \
    --mesh-dataset AltDataset \
    --epochs 500 \
    --device cuda:0 \
    --matcap-path data/matcap/rainbow.png \
    --validation 1