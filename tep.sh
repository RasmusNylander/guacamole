MODELS="resnet18 resnet152"
QUEUE="gpua100"

for MODEL in $MODELS
do
	
	n="$MODEL-$QUEUE"
	
	config="#!/bin/bash

	#BSUB -q $QUEUE
	#BSUB -J $n
	#BSUB -o outs/$n_%J.out
	#BSUB -n 1
	#BSUB -R "rusage[mem=10GB]"
	#BSUB -W 06:00
	#BSUB -gpu "num=1:mode=exclusive_process"

	module load python3/3.11.3
	module load cuda/12.1.1
	source ~/02514/projects/venv/bin/activate
	"
	

	command="python train.py --name=$n --model=$MODEL --logging=True --epochs=30 --batch_size=128"
	echo "$config$command" | bsub
	
done