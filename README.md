# Explainable Natural Language to Bash Translation using Abstract Syntax Tree

## Requirements

To install requirements:

```setup
conda env create -f environment. yml
```

## Training

To train the model in the paper, run this command:

```train
python src/main.py --mode train --batch 10 --accumulate_grad_batches 50
```

## Predictions

To generate results on test data, run:

```predictions
python src/main.py --mode predict --checkpoint_path path_to_chkpt 
```