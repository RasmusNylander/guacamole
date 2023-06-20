MODELS="resnet18 resnet152"

for MODEL in $MODELS
do
	
	n="$MODEL"
	
	config="#!/bin/bash

	#BSUB -q gpua40
	#BSUB -J $n
	#BSUB -o outs/$n_%J.out
	#BSUB -n 1
	#BSUB -R "rusage[mem=20GB]"
	#BSUB -W 04:00
	#BSUB -gpu "num=1:mode=exclusive_process"

	module load python3/3.11.3
	module load cuda/12.1.1
	source ~/02514/projects/venv/bin/activate
	"
	

	command="python train.py --name=$n --model=$MODEL --logging=True --epochs=30"
	echo "$config$command" | bsub
	
done