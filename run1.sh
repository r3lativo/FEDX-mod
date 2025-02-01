#!/bin/sh
python3 main.py --batch-size 256 --epochs 5 --comm_round 20 --n_parties 5 --portion 0.1 --device cuda:1 --method default --dataset cifar10 --rel_loss
python3 main.py --batch-size 256 --epochs 5 --comm_round 20 --n_parties 5 --portion 0.1 --device cuda:1 --method default --dataset svhn --rel_loss
python3 main.py --batch-size 256 --epochs 5 --comm_round 20 --n_parties 5 --portion 0.1 --device cuda:1 --method simsiam --dataset cifar10 --rel_loss
python3 main.py --batch-size 256 --epochs 5 --comm_round 20 --n_parties 5 --portion 0.1 --device cuda:1 --method simsiam --dataset svhn --rel_loss