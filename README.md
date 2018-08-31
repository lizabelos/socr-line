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

If you want to enable test during the training, use the command line argument ```--icdartest```.

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

## How to create a custom dataset

```python

class MyCustomDataset(Dataset):

    def __init__(self, path, loss=None):
        self.loss = loss
        ...

    def __getitem__(self, index):
        image_path, regions = self.labels[index % len(self.labels)]

	image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        
        ...
        
	label = self.loss.document_to_ytrue(np.array([witdth, height], dtype='int32'), np.array(regions, dtype='int32'))

        image = np.array(image, dtype='float') / 255.0

        return torch.from_numpy(image), torch.from_numpy(label)

    def __len__(self):
        return len(self.labels)
```

## How to create a custom model

Just like a normal Pytorch model : 

```python

class MyCustomModel(torch.nn.Module):

    def __init__(self):
        super(dhSegment, self).__init__()

	self.conv = torch.nn.Conv2d(3, 2, kernel_size=7, padding=3, stride=2, bias=False)
  
    def forward(self, input):
        input = self.conv(input)
        return input

    def create_loss(self):
        return MyCustomLoss()

```

## How to create a custom loss

```python

class XHeightCCLoss(torch.nn.Module):
    """An absolute position Loss"""

    def __init__(self):
        """

        :param s: grid division, assuming we have only 1 bounding box per cell
        """
        super().__init__()

	self.mse = torch.nn.MSELoss()
        self.decoder = BaselineDecoder()
        self.encoder = BaselineEncoder()

    def forward(self, predicted, y_true):
        predicted = predicted.permute(1, 0, 2, 3).contiguous()
        y_true = y_true.permute(1, 0, 2, 3).contiguous()

        return self.mse(predicted, y_true)

    def document_to_ytrue(self, image_size, base_lines):
        return self.encoder.encode(image_size, base_lines)

    def ytrue_to_lines(self, image, predicted, with_images=True):
        return self.decoder.decode(image, predicted, with_images, degree=3, brut_points=True)
```

## Generated document

Use the ```--generated``` argument to use generate document with ICDAR.

[Scribbler](https://github.com/dtidmarsh/scribbler)

To generate document with handwritten text, you will need to download the IAM dataset from here : [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database). At the initialization, please call init_iam_handwriting_line_dataset from scribbler.ressources.ressources_helper with the path of IAM dataset".