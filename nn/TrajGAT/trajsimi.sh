#!/bin/bash

exe_file="./main.py"
seeds=(2000 2001 2002 2003 2004)

function evalcmd () {
    echo $1
    eval $1
    sleep 0.5s
}

dataset=$1 # porto_20200 didi_20200
measurement=$2 # edr edwp hausdorff frechet
seeds=(2000 2001 2002 2003 2004)

for (( c=1; c<=5; c++ ))
do
    wholecommand="python ./main.py --config=model_config_${dataset}.yaml --dis_type=${measurement} --seed=${seeds[$c-1]}"
    evalcmd "$wholecommand"
done
