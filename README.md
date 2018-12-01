# Few-shot learning

The aim for this repository is to contain clean, readable and tested
code to reproduce few-shot learning research.

This project is written in python 3.6 and Pytorch and assumes you have
a GPU.

See these Medium articles for some more information
1. [Theory and concepts](https://towardsdatascience.com/advances-in-few-shot-learning-a-guided-tour-36bc10a68b77)
2. [Discussion of implementation details](https://towardsdatascience.com/advances-in-few-shot-learning-reproducing-results-in-pytorch-aba70dee541d)

# Setup
### Requirements

Listed in `requirements.txt`. Install with `pip install -r
requirements.txt` preferably in a virtualenv.

### Data
Edit the `DATA_PATH` variable in `config.py` to the location where
you store the Omniglot and miniImagenet datasets.

After acquiring the
data and running the setup scripts your folder structure should look
like
```
DATA_PATH/
    Omniglot/
        images_background/
        images_evaluation/
    miniImageNet/
        images_background/
        images_evaluation/
```

**Omniglot** dataset. Download from https://github.com/brendenlake/omniglot/tree/master/python,
place the extracted files into `DATA_PATH/Omniglot_Raw` and run
`scripts/prepare_omniglot.py`

**miniImageNet** dataset. Download files from
https://drive.google.com/file/d/0B3Irx3uQNoBMQ1FlNXJsZUdYWEE/view,
place in `data/miniImageNet/images` and run `scripts/prepare_mini_imagenet.py`

### Tests (optional)

After adding the datasets run `pytest` in the root directory to run
all tests.

# Results

The file `experiments/experiments.txt` contains the hyperparameters I
used to obtain the results given below.

### Prototypical Networks

![Prototypical Networks](https://github.com/oscarknagg/few-shot/blob/master/assets/proto_nets_diagram.png)


Run `experiments/proto_nets.py` to reproduce results from [Prototpyical
Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf)
(Snell et al).

**Arguments**
- dataset: {'omniglot', 'miniImageNet'}. Whether to use the Omniglot
    or miniImagenet dataset
- distance: {'l2', 'cosine'}. Which distance metric to use
- n-train: Support samples per class for training tasks
- n-test: Support samples per class for validation tasks
- k-train: Number of classes in training tasks
- k-test: Number of classes in validation tasks
- q-train: Query samples per class for training tasks
- q-test: Query samples per class for validation tasks


|                  | Omniglot |     |      |      |
|------------------|----------|-----|------|------|
| **k-way**        | **5**    |**5**|**20**|**20**|
| **n-shot**       | **1**    |**5**|**1** |**5** |
| Published        | 98.8     |99.7 |96.0  |98.9  |
| This Repo        | 98.2     |99.4 |95.8  |98.6  |

|                  | miniImageNet|     |
|------------------|-------------|-----|
| **k-way**        | **5**       |**5**|
| **n-shot**       | **1**       |**5**|
| Published        | 49.4        |68.2 |
| This Repo        | 48.0        |66.2 |

### Matching Networks

A differentiable nearest neighbours classifier.

![Matching Networks](https://github.com/oscarknagg/few-shot/blob/master/assets/matching_nets_diagram.png)

Run `experiments/matching_nets.py` to reproduce results from [Matching
Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf)
(Vinyals et al).

**Arguments**
- dataset: {'omniglot', 'miniImageNet'}. Whether to use the Omniglot
    or miniImagenet dataset
- distance: {'l2', 'cosine'}. Which distance metric to use
- n-train: Support samples per class for training tasks
- n-test: Support samples per class for validation tasks
- k-train: Number of classes in training tasks
- k-test: Number of classes in validation tasks
- q-train: Query samples per class for training tasks
- q-test: Query samples per class for validation tasks
- fce: Whether (True) or not (False) to use full context embeddings (FCE)
- lstm-layers: Number of LSTM layers to use in the support set
    FCE
- unrolling-steps: Number of unrolling steps to use when calculating FCE
    of the query sample

I had trouble reproducing the results of this paper using the cosine
distance metric as I found the converge to be slow and final performance
dependent on the random initialisation. However I was able to reproduce
(and slightly exceed) the results of this paper using the l2 distance
metric.

|                     | Omniglot|     |      |      |
|---------------------|---------|-----|------|------|
| **k-way**           | **5**   |**5**|**20**|**20**|
| **n-shot**          | **1**   |**5**|**1** |**5** |
| Published (cosine)  | 98.1    |98.9 |93.8  |98.5  |
| This Repo (cosine)  | 92.0    |93.2 |75.6  |77.8  |
| This Repo (l2)      | 98.3    |99.8 |92.8  |97.8   |

|                        | miniImageNet|     |
|------------------------|-------------|-----|
| **k-way**              | **5**       |**5**|
| **n-shot**             | **1**       |**5**|
| Published (cosine, FCE)| 44.2        |57.0 |
| This Repo (cosine, FCE)| 42.8        |53.6 |
| This Repo (l2)         | 46.0        |58.4 |

### Model-Agnostic Meta-Learning (MAML)

![MAML](https://github.com/oscarknagg/few-shot/blob/master/assets/maml_diagram.png)

I used max pooling instead of strided convolutions in order to be
consistent with the other papers. The miniImageNet experiments using
2nd order MAML took me over a day to run.

Run `experiments/maml.py` to reproduce results from [Model-Agnostic
Meta-Learning](https://arxiv.org/pdf/1703.03400.pdf)
(Finn et al).

**Arguments**
- dataset: {'omniglot', 'miniImageNet'}. Whether to use the Omniglot
    or miniImagenet dataset
- distance: {'l2', 'cosine'}. Which distance metric to use
- n: Support samples per class for few-shot tasks
- k: Number of classes in training tasks
- q: Query samples per class for training tasks
- inner-train-steps: Number of inner-loop updates to perform on training
    tasks
- inner-val-steps: Number of inner-loop updates to perform on validation
    tasks
- inner-lr: Learning rate to use for inner-loop updates
- meta-lr: Learning rate to use when updating the meta-learner weights
- meta-batch-size: Number of tasks per meta-batch
- order: Whether to use 1st or 2nd order MAML
- epochs: Number of training epochs
- epoch-len: Meta-batches per epoch
- eval-batches: Number of meta-batches to use when evaluating the model
    after each epoch


NB: For MAML n, k and q are fixed between train and test. You may need
to adjust meta-batch-size to fit your GPU. 2nd order MAML uses a _lot_
more memory.

|                  | Omniglot |     |      |      |
|------------------|----------|-----|------|------|
| **k-way**        | **5**    |**5**|**20**|**20**|
| **n-shot**       | **1**    |**5**|**1** |**5** |
| Published        | 98.7     |99.9 |95.8  |98.9  |
| This Repo (1)    | 95.5     |99.5 |92.2  |97.7  |
| This Repo (2)    | 98.1     |99.8 |91.6  |95.9  |

|                  | miniImageNet|     |
|------------------|-------------|-----|
| **k-way**        | **5**       |**5**|
| **n-shot**       | **1**       |**5**|
| Published        | 48.1        |63.2 |
| This Repo (1)    | 46.4        |63.3 |
| This Repo (2)    | 47.5        |64.7 |

Number in brackets indicates 1st or 2nd order MAML.