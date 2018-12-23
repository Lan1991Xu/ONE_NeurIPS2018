# Knowledge Distillation by On the fly Native Ensemble （ONE）NeurIPS2018
This is an [Pytorch](https://pytorch.org) implementation of  [Xu et al. Knowledge Distillation On the Fly Native Ensemble (ONE) NeurIPS 2018](https://arxiv.org/pdf/1806.04606.pdf) on Python 2.7, Pytorch 2.0. 
You may refer to our [Vedio](http://www.eecs.qmul.ac.uk/~xl309/Doc/Publication/2018/NIPS2018/ONE-Slide-PPT.mp4) and [Poster](http://www.eecs.qmul.ac.uk/~xl309/Doc/Publication/2018/NIPS2018/Poster_landscape.pdf)  for a quick overview.

# ONE





## Getting Started

### Prerequisites:

- Datasets: CIFAR100, CIFAR10
- Python 2.7. 
- Pytorch version == 2.0.0. 




## Running Experiments

### Training: 


For example, to train the ONE model using `ResNet-32` on CIFAR100, run the the following scripts.
```
bash ONE_ResNet32.sh
```
to train baseline model using `ResNet-32` on CIFAR100, run the the following scripts.
```
python cifar_baseline.py -a resnet --dataset cifar100 --depth 32 --epochs 300  --gpu-id 5 --schedule 151 225  --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/baseline-32
```
ResNet-110 can also be executed by change the depth 32 to 110

## Tip for Stabilizing Model Training
It may help to ramp up [https://arxiv.org/abs/1703.01780] the KL cost in the beginning over the first few epochs until the teacher network starts giving good predictions.
## Citation
Please refer to the following if this repository is useful for your research.

### Bibtex:

```
@inproceedings{lan2018knowledge,
  title={Knowledge Distillation by On-the-Fly Native Ensemble},
  author={Lan, Xu and Zhu, Xiatian and Gong, Shaogang},
  booktitle={Advances in Neural Information Processing Systems},
  pages={7527--7537},
  year={2018}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


## Acknowledgements

This repository is partially built upon the [bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification) repository. 
