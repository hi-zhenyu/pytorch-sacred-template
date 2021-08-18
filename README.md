# pytorch-sacred-template
A general template for deep learning based on PyTorch and Sacred.

This repository takes training a classifier on for digit recognition (MNIST) as an example.


## Why PyTorch

[PyTorch](https://pytorch.org/) is an open source machine learning framework that accelerates the path from research prototyping to production deployment. 

> More Pythonic, Easy to Use, Useful Libraries, Effortless Data Parallelism, Excellent for Researchers ... 

> See [Reasons to Choose PyTorch for Deep Learning](https://towardsdatascience.com/reasons-to-choose-pytorch-for-deep-learning-c087e031eaca) for more details.



## Why Sacred

[Sacred](https://github.com/IDSIA/sacred) is a tool to help you configure, organize, log and reproduce experiments developed at IDSIA.

We suggest it for experiment management as it supports many features:

1. Automatic code versioning for reproducibility

2. Easily experiment config saving, logging, and outputs. (Sacred Observer)

3. Various Frontend supporting


## Quickstart


### Requirements

Install PyTorch and Sacred.

```
pip install -r requirements.txt
```

> We suggest PyTorch>=1.6 for automatic mixed precision (AMP) feature.

We assume you are already familiar to PyTorch, so let us quickly go through the basic [Sacred tutorial](https://sacred.readthedocs.io/en/stable/quickstart.html) and click [this](https://github.com/hi-zhenyu/pytorch-sacred-template/generate) to clone this repository to your github.

Next, we will run a simple example to see what this template could do.

### A simple example

A simple example of training a classifier for digit recognition (MNIST):

```
python run.py
```

The we got the outputs:

```
INFO - pytorch - Running command 'run'
INFO - pytorch - Started run with ID "1"
INFO - root - 
 *--------------- Experiment Config ---------------*
INFO - root - Namespace(batch_size=128, data_dir='./data', learning_rate=0.001, log_step=100, num_classes=10, num_epochs=1, output_dir='./output/1', output_root='./output', val_split=0.01)

*--------------- Training ---------------*

Epoch 1/1
Training[  0/465]	b_t  0.20 ( 0.20)	d_t  0.01 ( 0.01)	loss 2.3008e+00 (2.3008e+00)
Training[100/465]	b_t  0.01 ( 0.01)	d_t  0.01 ( 0.01)	loss 2.8045e-01 (5.9409e-01)
Training[200/465]	b_t  0.01 ( 0.01)	d_t  0.01 ( 0.01)	loss 1.2773e-01 (3.6994e-01)
Training[300/465]	b_t  0.01 ( 0.01)	d_t  0.01 ( 0.01)	loss 2.0513e-01 (2.7966e-01)
Training[400/465]	b_t  0.01 ( 0.01)	d_t  0.01 ( 0.01)	loss 7.0957e-02 (2.3148e-01)
Validation
Validation[0/5]	b_t  0.01 ( 0.01)	d_t  0.01 ( 0.01)	loss 4.6632e-02 (4.6632e-02)
acc 0.9916666666666667

*--------------- Testing ---------------*
Testing[ 0/79]	b_t  0.01 ( 0.01)	d_t  0.01 ( 0.01)
ACC 0.9839
INFO - pytorch - Result: 0.9839
INFO - pytorch - Completed after 0:00:06
```

The checkpoints, experiment config, recorded metrics, enviornment info and all outputs will be saved to stdout and stderr will be saved to ```./output/1```.

The source code will backup in ./output/_sources

```

├── output
│   ├── 1                                                       # Experiment ID
│   │   ├── checkpoint_1.pt                                     # Saved checkpoint
│   │   ├── config.json                                         # Saved experiment config
│   │   ├── cout.txt                                            # Experiment output (stdout and stderr)
│   │   ├── metrics.json                                        # Recorded experiment metrics
│   │   ├── model_best.pt                                       # Best checkpoint on validation data
│   │   └── run.json                                            # Experiment info (env, code, gpu&cpu, etc.)
│   └── _sources
│       ├── main_2a180b6f96d317238aac22fcf5d07dfa.py            # backup code
│       ├── network_4790c945aa7e76e2a6dd58e5038218dd.py         # backup code
│       └── run_a041462fa92b192a8aec23efcca8a38b.py             # backup code

```

## File structure

```

run.py               # The entry point. (parameter and environment settings, Sacred config)
network.py           # Network design (networks, loss functions)
main.py              # Training and testing scripts
utils.py             # Some convenience functions
data.py              # Data modules

```

## Contributions

Suggestions and useful tricks are welcome!