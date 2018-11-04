# Few-shot learning

The aim for this repository is to contain clean, readable code to
reproduce few-shot learning research.

This project is written in python 3.6 and Pytorch.

- [x] Reproduce Prototypical Networks to a few % on Omniglot
- [x] Reproduce Prototypical Networks to a few % on miniImageNet
- [x] Reproduce Matching Networks to a few % on Omniglot
- [ ] Reproduce Matching Networks to a few % on miniImageNet
- [x] Correctly implement FCE for Matching Networks
- [ ] Upload pretrained models
- [ ] Clean up code
- [ ] Prettify README
- [ ] Write blog post

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
### Prototypical Networks

Run `experiments/proto_nets.py` to reproduce results from [Prototpyical
Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf)
(Snell et al).

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

Run `experiments/matching_nets.py` to reproduce results from [Matching
Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf)
(Vinayls et al).

I had trouble reproducing the results of this paper using a cosine
distance metric but was able to reach similar performace using the L2
distance metric.

|                     | Omniglot|     |      |      |
|---------------------|---------|-----|------|------|
| **k-way**           | **5**   |**5**|**20**|**20**|
| **n-shot**          | **1**   |**5**|**1** |**5** |
| Published (cosine)  | 98.1    |98.9 |93.8  |98.5  |
| This Repo (cosine)  | 92.0    |92.8 |75.6  |77.8  |
| This Repo (l2)      | 98.3    |99.8 |92.8  |98.2   |

|                     | miniImageNet|     |
|---------------------|-------------|-----|
| **k-way**           | **5**       |**5**|
| **n-shot**          | **1**       |**5**|
| Published (cosine)  | 44.2        |57.0 |
| This Repo (cosine)  | 41.3        |48.8 |
| This Repo (l2)      | 42.4        |54.5 |