# reid_baseline

## Train
Train a model by
```bash
python3 main.py train --dataset_name Market1501 --model ResNet50 
```


## Test
Use trained model to extract feature by
```bash
python main.py test --dataset_name Market1501 --model ResNet50
```
