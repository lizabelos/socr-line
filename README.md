# SOCR LINE

## Requirements

 - Pytorch 0.4 : ```conda install pytorch=0.4 -c pytorch```
 - OpenCV 3 : ```conda install opencv```
 - scikit-image :  ```conda install -c conda-forge scikit-image```
 - All in requirements.txt ```pip instal -r requirements.txt'```

## Compilation

```
python3 setup.py build_ext --inplace
```

## Training

```
python3 train.py --icdartrain [train_path]
```

## Evaluate

```
python3 evaluate.py path
```

## Dataset

This is the link to ICDAR Complex Dataset :

[ICDAR cBAD 2017](https://scriptnet.iit.demokritos.gr/competitions/5/1/)

You you want to enable test during the training, you have to split yourself the dataset into a train part and a test part.