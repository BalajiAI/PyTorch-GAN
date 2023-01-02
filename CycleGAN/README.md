# Image-to-Image Translation using CycleGAN
### Problem statement
The Problem is to convert an image from a Source Domain A to Target Domain B. For example, Let's say that you've an image which is taken at Summer and you need to convert it to an image as if it were taken at Winter. But to do this, we need a training set of aligned image pairs, which means that you need to have images taken at exact same location and at both Summer and Winter. However, for many tasks, paired training data will not be available.

![Season Transfer](https://junyanz.github.io/CycleGAN/images/season.jpg)

CycleGAN solves this problem  of paired training data by introducing an additional loss called Cycle consistent loss along with Adversarial loss for training. So the name, CycleGAN.

### Summer2Winter Dataset

The Dataset for training CycleGAN can be downloaded using [this link](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/).

### Training
You can able to train by executing **train.py**.

### Training procedure
You can easily able to understand the Training procedure by looking at the below image.
![Training procedure](https://blog.jaysinha.me/content/images/size/w2000/2021/03/cyclegan.png)

## CycleGAN paper
### Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks by Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros
You can read the original paper [here](https://arxiv.org/abs/1703.10593).

```
@misc{zhu2020unpaired,
      title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks}, 
      author={Jun-Yan Zhu and Taesung Park and Phillip Isola and Alexei A. Efros},
      year={2020},
      eprint={1703.10593},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
