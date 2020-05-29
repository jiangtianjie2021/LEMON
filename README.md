# Deep Learning Library Testing via Effective Model Generation

This is the implement repository of our upcoming ESEC/FSE 2020 paper:  **Deep Learning Library Testing via Effective Model Generation.** 

## Description

`LEMON` is a novel approach to test DL libraries by generating effective DL models via guided mutation whose goal is to test DL libraries as sufficiently as possible by exploring unused library code or different usage ways of library code. We further propose a heuristic strategy in `LEMON` to guide the process of model generation so as to generate models that can amplify the inconsistent degrees for real bugs. In this way, it is clearer to distinguish real bugs and uncertain impacts in DL libraries. We conducted an empirical study to evaluate the effectiveness of `LEMON` based on `20` release versions of `TensorFlow`,`Theano,` `CNTK`, and `MXNet`. `LEMON` detected `24` new bugs in the latest release versions of these libraries. The results also demonstrate that the models generated by `LEMON` outperform existing models and the models generated without guidance in terms of the number of unique bugs/inconsistencies and the achieved inconsistent degrees. 

## Datasets/Models/Libraries

### Datasets/Models

We used `12` popular DL `models` based on `6` `datasets` including both images and sequential data, as the initial seed models in `LEMON`, which have been widely used in many existing studies.

| Model       | Dataset              | Related link<sup>1</sup>                                     |
| ----------- | -------------------- | ------------------------------------------------------------ |
| AlexNet     | CIFAR-10             | [alexnet-cifar-10-keras-jupyter](https://github.com/toxtli/alexnet-cifar-10-keras-jupyter/blob/master/alexnet_test1.ipynb) |
| LeNet5      | Fashion-MNIST        | [fashion_mnist_keras](https://colab.research.google.com/github/margaretmz/deep-learning/blob/master/fashion_mnist_keras.ipynb) |
| LeNet5      | MNIST                | [lenet5-mnist](https://github.com/lucaaslb/lenet5-mnist)     |
| LSTM-1      | Sine-Wave            | [LSTM-Neural-Network-for-Time-Series-Prediction](https://github.com/StevenZxy/CIS400/tree/f69489c0624157ae86b5d8ddb1fa99c89a927256/code/LSTM-Neural-Network-for-Time-Series-Prediction-master) |
| LSTM-2      | Stock-Price          | [StockPricesPredictionProject](https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/StockPricesPredictionProject) |
| ResNet50    | ImageNet<sup>2</sup> | Keras applications tutorial<sup>3</sup>                      |
| MobileNetV1 | ImageNet<sup>2</sup> | Keras applications tutorial<sup>3</sup>                      |
| InceptionV3 | ImageNet<sup>2</sup> | Keras applications tutorial<sup>3</sup>                      |
| DenseNet121 | ImageNet<sup>2</sup> | Keras applications tutorial<sup>3</sup>                      |
| VGG16       | ImageNet<sup>2</sup> | Keras applications tutorial<sup>3</sup>                      |
| VGG19       | ImageNet<sup>2</sup> | Keras applications tutorial<sup>3</sup>                      |
| Xception    | ImageNet<sup>2</sup> | Keras applications tutorial<sup>3</sup>                      |

1:  The first 5 models are trained using existing repositories while the last 7 models in ImageNet are obtained directly using the API provided by Keras.

2: We sampled 1500 images from ImageNet and you can also sample your own images from the [ImageNet validation dataset](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads). 

3: Keras applications tutorial can be found in: https://keras.io/api/applications/

4: All model files and two regression dataset and ImageNet dataset we sampled can be access in [OneDrive](https://1drv.ms/u/s!Aj6dGBsJFcs0jnXVUfAtsEjdUW_T?e=ezo32C)


### Libraries

We used `20` release versions of  `4` widely-used DL `libraries`, i.e., `TensorFlow`, `CNTK`,`Theano`, and `MXNet`, as subjects to constructed five experiments (indexed `E1` to `E5` in Table) to conduct differential testing.

We share the link of each library and docker image used in `LEMON`. 

| Experiment ID | Tensorflow                                                | Theano                                          | CNTK                                                         | MXNet                                                      | CUDA                                                         |
| ------------- | --------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- | ------------------------------------------------------------ |
| E1            | [1.14.0](https://pypi.org/project/tensorflow-gpu/1.14.0/) | [1.0.4](https://pypi.org/project/Theano/1.0.4/) | [2.7.0](https://pypi.org/project/cntk-gpu/2.7/)              | [1.5.1](https://pypi.org/project/mxnet-cu101/1.5.1.post0/) | [10.1](https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=10.1-cudnn7-devel-ubuntu16.04) |
| E2            | [1.13.1](https://pypi.org/project/tensorflow-gpu/1.13.1/) | [1.0.3](https://pypi.org/project/Theano/1.0.3/) | [2.6.0](https://pypi.org/project/cntk-gpu/2.6/)              | [1.4.1](https://pypi.org/project/mxnet-cu100/1.4.1/)       | [10.0](https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=10.0-cudnn7-devel-ubuntu16.04) |
| E3            | [1.12.0](https://pypi.org/project/tensorflow-gpu/1.12.0/) | [1.0.2](https://pypi.org/project/Theano/1.0.2/) | [2.5.1](https://pypi.org/project/cntk-gpu/2.5.1/)            | [1.3.1](https://pypi.org/project/mxnet-cu90/1.3.1/)        | [9.0](https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=9.0-cudnn7-devel-ubuntu16.04) |
| E4            | [1.11.0](https://pypi.org/project/tensorflow-gpu/1.11.0/) | [1.0.1](https://pypi.org/project/Theano/1.0.1/) | [2.4.0](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Linux-Python?tabs=cntkpy24) | [1.2.1](https://pypi.org/project/mxnet-cu90/1.2.1.post1/)  | [9.0](https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=9.0-cudnn7-devel-ubuntu16.04) |
| E5            | [1.10.0](https://pypi.org/project/tensorflow-gpu/1.10.0/) | [1.0.0](https://pypi.org/project/Theano/1.0.0/) | [2.3.1](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Linux-Python?tabs=cntkpy231) | [1.1.0](https://pypi.org/project/mxnet-cu90/1.1.0/)        | [9.0](https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=9.0-cudnn7-devel-ubuntu16.04) |

\* All libraries should be  `GPU-supported` version

## Reproducibility

### Environment 

We conducted 5 experiments in `LEMON` of which the library and CUDA version information are as described above. In order to facilitate other researchers to reproduce `LEMON`, we provide a `docker`  image for the `E1` experiment. It can be easily obtained by the following command. **Note: [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker) is required. **

**Step 0:** Install `nvidia-docker2`. You can use this [instruction](https://codepyre.com/2019/01/installing-nvidia-docker2-on-ubuntu-18.0.4/) to install it.

**Step 1: ** Using the following command to pull the docker image we released for `E1` and create a container for it.

> docker pull yenming1227/lemon:e1
>
> docker run --runtime=nvidia -it -v /your/local/path/:/data --name "lemon_exp01" yenming1227/lemon:e1 /bin/bash

Then you will enter a container.

**Note: If your server is using http proxy, you should configure proxy in the container just as you did in your server before**

**Step 2:**  Create five virtual environments as shown below in your docker container.

 ```shell
# tensorflow
conda create -n tensorflow python=3.6
source activate tensorflow
pip install -r lemon_requirements.txt
pip install keras==2.2.4
pip install tensorflow-gpu==1.14.0
source deactivate

# theano
conda create -n theano python=3.6
source activate theano
pip install -r lemon_requirements.txt
conda install pygpu=0.7.6
pip install keras==2.2.4
pip install theano==1.0.4
source deactivate

# cntk
conda create -n cntk python=3.6
source activate cntk
pip install -r lemon_requirements.txt
pip install keras==2.2.4
pip install cntk-gpu==2.7
source deactivate

# mxnet
conda create -n mxnet python=3.6
source activate mxnet
pip install -r lemon_requirements.txt
pip install keras-mxnet==2.2.4.2
pip install mxnet-cu101==1.5.1.post0
source deactivate

# default lemon python
conda create -n lemon python=3.6
source activate lemon
pip install -r lemon_requirements.txt
pip install keras==2.2.4
pip install tensorflow-gpu==1.14.0
source deactivate
 ```

### Redis Startup

LEMON uses redis to store intermediate outputs and exchange data between different processes. We have installed redis in our docker image, you can start it with the following command:

> cd  /root/redis-4.0.8/src
>
> ./redis-server ../redis.conf

### Configuration

Before running, please configure the path used in `LEMON`. All options are defined in `config/experiments.conf`

```
[parameters]
mutate_ops=WS GF NEB NAI NS ARem ARep LA LC LR LS MLA
metrics=D_MAD

# Initial Models and dataset 
exp= alexnet-cifar10 xception-imagenet lenet5-fashion-mnist lenet5-mnist resnet50-imagenet vgg16-imagenet vgg19-imagenet densenet121-imagenet mobilenet.1.00.224-imagenet inception.v3-imagenet lstm0-sinewave lstm2-price

# Path of the initial models
# Name model file as 'alexnet-cifar10_origin.h5'
origin_model_dir=/your/path/save_origin_model

# Path of the ImageNet and regression dataset
dataset_dir=/your/path/save_dataset

# Modifying the backends option is not recommended.
# There is some hard-code in the program about the backends
backend=tensorflow theano cntk mxnet

# Prefix of Anaconda virtual environment path 
python_prefix = /root/anaconda3/envs/

# Path to save results
output_dir = /your/path/save_experiment_result

# Number of mutant
mutate_num=100

# Number of inputs(maximum 1500)
test_size=1500

# Seed pool size
pool_size=50

# Mutate ratio
mutate_ratio=0.3

#GPU ID (At most 2 GPUs)
gpu_ids = 0,1

# Inconsistency threshold
threshold = 0.4

[redis]
host= 127.0.0.1 # your-redis-server
port= 6379 # redis port
redis_db= 0 # db number

```

### Running LEMON

The `LEMON` artifacts are well organized, and researchers can simply run `LEMON` with the following command (all commands are executed in `LEMON` root path)

**Note: Since we conducted five large scale experiments (generating 100 mutants for each of the 12 initial models and analyzing inconsistencies on 1500 inputs and locating bugs), it could not be completed within 48  hours. Therefore, we provide a demo run, which can be completed within 1 hour.**

> source activate lemon

**Mutation:**

> python -u -m run.mutation_executor demo.conf

The above command shows how to generate mutants and calculating inconsistencies in `LEMON`. `demo.conf` is the configuration file we provided for `demo run`. Of course you should configure the self-defined path information (e.g. dataset_dir ) in it as described in `experiments.conf`.

**Localization:**

> python -u -m run.localization_executor demo.conf

This command shows the way to perform localization in `LEMON`. The final  bug reports will be stored in path `/your/path/save_experiment_result/bug_list.txt` 

In this `demo run`, you may get `6` suspected bugs in  `bug_list.txt`. If you want to reproduce all bugs reported in paper (exclude 1 Keras performance bug), you should configure  the path in `experiments.conf` and run with it.

### Extension

`LEMON` also supports researchers to switch to other models and datasets. You only need to focus on the code snippets of the data processing part in `DataUtils.get_data_by_exp` in `scripts/tools/utils/py`.

```
# TODO: Add your own data preprocessing here
# Note: The returned inputs should be preprocessed and labels should decoded as one-hot vectors which could be directly feed in model. Both of them should be returned in batch, e.g. shape like (1500,28,28,1) and (1500,10)
# 
# elif 'xxx' in exp:
#     x_test, y_test = get_your_data(dataset_dir)
```

Besides, you should name your model file in format ` NetworkName-DatasetName_origin.h5`, e.g.  `mobilenet.1.00.224-imagenet_origin.h5`. 

Note: `_` and `-` can't be shown in `NetworkName`. You can replace them with `.` 

For example , changing  `mobilenet_1.00_224-imagenet_origin.h5` to `mobilenet.1.00.224-imagenet_origin.h5`. 

## Contact

Authors information:

| Name          | Email Address          | **Github id** |
| ------------- | ---------------------- | ------------- |
| Zan Wang      | wangzan@tju.edu.cn     | tjuwangzan    |
| Ming Yan      | yanming@tju.edu.cn     | jacob-yen     |
| Junjie Chen * | junjiechen@tju.edu.cn  | JunjieChen    |
| Shuang Liu    | shuang.liu@tju.edu.cn  | AbigailLiu    |
| Dongdi Zhang  | zhangdongdi@tju.edu.cn | Dandy-John    |

\* *corresponding author*

