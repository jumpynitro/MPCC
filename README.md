# Generative adversarial network for clustering. 

## MPCC: Matching Priors and Conditionals for Clustering
	
Official implemenation of [MPCC: Matching Priors and conditionals for Clustering](https://arxiv.org/abs/2008.09641). This respository is strongly based on ''The author's officially unofficial PyTorch BigGAN'' [implementation](https://github.com/ajbrock/BigGAN-PyTorch).

## How To Use This Code
You will need:

- [PyTorch](https://PyTorch.org/), version 1.2 (Although newer versions are also posible to use)
- matplotlib, tqdm, numpy and scipy

Note that the official FID score and IS are based on tensorflow implementations. You will need tensorflow 1.1 and 1.3 respectively to obtain these official metrics using `inception_tf13_p.py` and `fid_p.py`. You can find C10 inception metrics in [here](https://drive.google.com/file/d/1TjwetbupClCWxDUtA8qcDFK1lxzf4a90/view?usp=sharing).

A jupyter notebook is provided to perform generation, reconstructions and predictions with MPCC. Additionally Cifar10 models weights are [included](https://drive.google.com/file/d/1TjwetbupClCWxDUtA8qcDFK1lxzf4a90/view?usp=sharing).

Here are some samples of the generative model:

Cifar10 samples (every two columns a different cluster):

![Cifar10 samples](imgs/samples_cifar.jpg?raw=true "Cifar10 samples")


Cifar20 samples (every row a different cluster):

![Cifar20 samples](imgs/Add_samples_C20.jpg?raw=true "Cifar20 samples")

Omniglot samples (every row a different cluster): 

![Omniglot samples](imgs/Add_samplesOmni.jpg?raw=true "Omniglot samples")

