# Knowledge Distillation On the Fly Native Ensemble （NeurIPS2018）
This is an [Pytorch](https://pytorch.org) implementation of  [Xu et al. Knowledge Distillation On the Fly Native Ensemble (ONE) NeurIPS 2018](https://arxiv.org/pdf/1806.04606.pdf) on Python 2.7, Pytorch 2.0. 
You may refer to our [vedio](http://www.eecs.qmul.ac.uk/~xl309/Doc/Publication/2018/NIPS2018/ONE-Slide-PPT.mp4) and [poster](http://www.eecs.qmul.ac.uk/~xl309/Doc/Publication/2018/NIPS2018/Poster_landscape.pdf)  for a quick overview.

# ONE





## Getting Started

### Prerequisites:

- Datasets: CIFAR100, CIFAR10
- Python 2.7. 
- Pytorch version == 2.0.0. 

```


## Running Experiments

### Training: 




For example, to train the DAL model using `mobilenet_v1` on MARS, run the the following scripts.

```
bash scripts/train_MARS.sh
```




### Testing: 


```

## Citation
Please refer to the following if this repository is useful for your research.

### Bibtex:

```
@article{lan2018knowledge,
  title={Knowledge Distillation by On-the-Fly Native Ensemble},
  author={Lan, Xu and Zhu, Xiatian and Gong, Shaogang},
  journal={arXiv preprint arXiv:1806.04606},
  year={2018}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


## Acknowledgements

This repository is partially built upon the [bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification) repository. The evaluation code (cmc & mAP) is partially borrowed from the [MARS-evaluation](https://github.com/liangzheng06/MARS-evaluation) repository. 
