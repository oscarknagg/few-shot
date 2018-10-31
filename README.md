# Few-shot learning

Repository containing code to reproduce few-shot learning research.

This project is written in python 3.6 and Pytorch.

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


# Results
### Prototypical Networks

Run `experiments/proto_nets.py` to reproduce results from

| Dataset           | Omniglot |     |     |     | miniImageNet|    |
|-------------------|----------|-----|-----|-----|-------------|-----|
| **k-way**         | **5**    |     |**2**|     | **5**       |     |
| **n-shot**        | **1**    |**5**|**1**|**5**| **1**       |**5**|
| ProtoNets (paper) | 98.8     |99.7 |96.0 |98.9 | 49.4        |68.2 |
| Protonets (repo)  | 95.6     |99.4 |87.8 |98.6 | 39.3        |53.8 |

### Matching Networks