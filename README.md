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
python3 train.py
```

## Evaluate

```
python3 evaluate.py path
```