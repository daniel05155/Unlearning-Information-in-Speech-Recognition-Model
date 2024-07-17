# Unlearning-Information-in-Speech-Recognition-Model

Nowadays,machine learning systems store a vast amount of training data. 
However, due to model inversion or membership inference attacks, certain private information from the training dataset may be exposed.
Although deleting data from back-end databases ought to be simple, doing so is insufficient when it comes to artificial intelligence (AI) because machine learning models frequently retain historical data. 
To address the above problem, we require a novel method known as machine unlearning, which enables deep learning models to forget about specific data following the training stage. 
Through the machine unlearing(MU) techniques, we are able to eliminate specific memories from the speech recognition model.

## Requirements
- Python 3.10
- PyTorch

### Install Environment
```
conda env create -f environment.yaml
```

## Usage

### Dataset
We use the [Google Speech Commands Dataset (v0.02)](https://arxiv.org/abs/1804.03209) as the training dataset.

### Commands
* Training
```
python main.py --epochs 50 --lr 0.01 --batch_size 256 
```

* Retrain
```
python main.py --retrain  --epochs 50
```

* Transfer learning
```
python unlearn_TS.py
```

* Uncertainty Enhancing
```
python advloss.py
```

* Inference
```
python inference.py
```

