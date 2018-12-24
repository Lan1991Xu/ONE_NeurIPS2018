python cifar_baseline.py -a resnet --dataset cifar100 --depth 110 --epochs 305 --schedule 151 225 --gamma 0.1 --wd 1e-4 --gpu-id 3 --checkpoint checkpoints/cifar100/Resnet-110-baseline-check
python cifar_one.py -a one_resnet --dataset cifar100 --depth 32 --epochs 300 --gpu-id 5 --schedule 151 225  --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/ONE-32
python cifar_one.py -a one_resnet --consistency_rampup 80 --dataset cifar100 --depth 32 --epochs 300 --gpu-id 5 --schedule 151 225  --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/ONE-32-rampup
python cifar_one.py -a one_resnet --consistency_rampup 80 --dataset cifar10 --depth 32 --epochs 300 --consistency_rampup 80 --gpu-id 3 --schedule 150 225  --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/ONE-32-rampup

python cifar_one.py -a one_resnet --consistency_rampup 80 --dataset cifar100 --depth 32 --epochs 300 --consistency_rampup 80 --gpu-id 5 --schedule 151 225  --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/ONE-32-rampup
python cifar_one.py -a one_resnet --consistency_rampup 80 --dataset cifar10 --depth 110 --epochs 300 --consistency_rampup 80 --gpu-id 6 --schedule 151 225  --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/ONE-110-rampup

python cifar_one.py -a one_resnet --dataset cifar100 --depth 110 --epochs 300 --gpu-id 6 --schedule 151 225  --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/ONE-110