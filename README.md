# The Kaggle's "What's Cooking?" competition.

The main information about the competition you can find [here](https://www.kaggle.com/c/whats-cooking)

1. Install Python's requirements

```bash
$ pip install -r requirements.txt
```

2. Train classifier

```bash
$ python train_model.py
```

3. Create submission file

```bash
$ python train_model.py --mode submission
```

To make your result reproducible you need to add ``--reproducible`` flag

```bash
$ python train_model.py --mode submission --reproducible
```

## Tests

To run unit tests you need to execute the following command in the terminal from project's root directory

```bash
$ nosetests
```

## Documentation

```
$ python train_model.py -h
usage: train_model.py [-h] [-m {validation,submission}] [-r]

optional arguments:
  -h, --help            show this help message and exit
  -m {validation,submission}, --mode {validation,submission}
                        set up training mode
  -r, --reproducible    make training reproducible
```
