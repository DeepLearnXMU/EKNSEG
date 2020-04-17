#!/bin/bash
data=nips
datapath=~/journal/data/entity/$data
kn=wordnet
graphpath=$datapath/$kn

batch=64
gnnl=3

if [ $data = nips || $data = aan ];then
batch=16
gnnl=2
h=300
fi

thre=
modelname=${data}
prefix=lower
nos=.nosingle
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --model ${modelname} --vocab $datapath/vocab.new.100d.${prefix}.pt \
    --corpus $datapath/train.${prefix} $datapath/train.eg.20${nos} $graphpath/train.eg.20${nos}.common \
    --valid $datapath/val.${prefix} $datapath/val.eg.20${nos} $graphpath/val.eg.20${nos}.common \
    --test $datapath/test.${prefix} $datapath/test.eg.20${nos} $graphpath/test.eg.20${nos}.common \
    --loss 0 --thre $thre \
    --writetrans decoding/${modelname}.devorder --ehid 150 --entityemb glove \
    --gnnl ${gnnl} --labeldim 50 --agg gate --gnndp 0.4 \
    --batch_size ${batch} --beam_size 64 --lr 1.0 --seed 1234 \
    --d_emb 100 --d_rnn 500 --d_mlp 500 --input_drop_ratio 0.5 --drop_ratio 0.5 \
    --save_every 1 --maximum_steps 100 --early_stop 5 >>${modelname}.train 2>&1 &

