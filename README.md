# SOCR LINE

## Requirements

 - Python3 with Cython
 - Pytorch 0.4 : ```conda install pytorch=0.4 -c pytorch```
 - OpenCV 3 : ```conda install opencv```
 - scikit-image :  ```conda install -c conda-forge scikit-image```
 - All in requirements.txt : ```pip install -r requirements.txt```

Recommended : 
 - Java (Runtime Environement) version 8 or more, to enable the [Transkribus Baseline Evaluation Scheme](https://github.com/Transkribus/TranskribusBaseLineEvaluationScheme).
 

## Compilation

SOCR Line was created with Cython. To compile it, run : 
```
python3 setup.py build_ext --inplace
```

## Training

To train the network, run : 
```
python3 train.py --icdartrain [train_path]
```

If you want to enable test during the training, use the commande line argument ```--icdartest```.

## Evaluate

To evaluate the network, where path is a directory or a image file, run : 
```
python3 evaluate.py path
```
The result file will be created in the ```result``` folder, in the ```socr-line``` directory.

## Dataset

This is the link to ICDAR Complex Dataset :

[ICDAR cBAD 2017](https://scriptnet.iit.demokritos.gr/competitions/5/1/)

You you want to enable test during the training, you have to split yourself the dataset into a train part and a test part.

## Generated document

You can also use Scribbler document generator from here [Scribbler](https://github.com/dtidmarsh/scribbler) by importing ```scribbler.generator.LineGeneratorSet```.

To generate document with handwritten text, you will need to download the IAM dataset from here : [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database). At the initialization, please call init_iam_handwriting_line_dataset from scribbler.ressources.ressources_helper with the path of IAM dataset".