# Library for Deep Residual Multiscale Segmenter (resmcseg)

[![Build Status](https://travis-ci.org/pybind/cmake_example.svg?branch=master)](https://travis-ci.org/pybind/cmake_example)
[![Build status](https://ci.appveyor.com/api/projects/status/57nnxfm4subeug43/branch/master?svg=true)](https://ci.appveyor.com/project/dean0x7d/cmake-example/branch/master)

The python library of Deep Residual Multiscale Segmenter (autonet). 
Current version just supports the KERAS package of deep learning and 
will extend to the others in the future. 

## Major modules

**model**

* gResMCSeg: major class to obtain a deep extensive residual multiscale  
      FCN. You can setup its aruments. See the class and its 
      member functions' help for details.  
* gResMCSegPre: major class to make semantic segmentation for binary and multi class. 
* pretrainedmodel: function to download the pretrained models using the DSTL and 
      ZURICH datasets from the Google cloud 

**util**

* segmetrics: main metrics including jaccard index, MIoU, and loss functions etc.
* helper: helper functions including color mapping etc.   

**data**

* data: function to access one image for Zurich to test the model's prediction. 
       
## Installation

You can directly install it using the following command for the latest version:

```bash
sudo pip install resmcseg
```

## Note for installation and use 

**Compiler requirements**

resmcseg requires a C++11 compliant compiler to be available.

**Runtime requirements**

resmcseg requires installation of Keras with support of Tensorflow as the 
backend system of deep learning (to support Keras). Also Pandas and Numpy should 
be installed. 
Specifics: Keras>=2.2.2; opencv2>=3.4.1,tensorflow,numpy,sklearn, pandas,
          gdal,tifffile etc.  

## Use case 
The homepage of the github for the package, resmcseg provides specific 
examples for use of the library:  
https://github.com/lspatial/resmcsegpub 

## License

The resmcseg is provided under a MIT license that can be found in the LICENSE
file. By using, distributing, or contributing to this project, you agree to the
terms and conditions of this license.

## Test call

```python
import os
import cv2
from keras.models import load_model,model_from_json
from resmcseg.util.segmetrics import compute_iou,jaccard,mean_iouC,miou,mean_iou
from resmcseg.model.gresmcseg_pre import gResMCSegPre
from resmcseg.model.resizelayer import ResizeLayer
from resmcseg.model.pretrainedmodel import downloadPretrainedModel
from resmcseg.util.helper import bce_dice_loss,jaccard_coef,jaccard_coef_int,jaccard_coef1
from resmcseg.util.helper import onehot_to_rgb,color_dict
from resmcseg.data import dload

modelFl='/tmp/model_strwei.h5'
if not os.path.isfile(modelFl):
    downloadPretrainedModel('ZURICH',destination=modelFl)
model = load_model(modelFl,custom_objects={'ResizeLayer': ResizeLayer,'bce_dice_loss':bce_dice_loss,
        'mean_iou':mean_iou,'jaccard_coef':jaccard_coef, 'jaccard_coef1':jaccard_coef1,'miou':miou,
        'jaccard_coef_int':jaccard_coef_int,'mean_iouC': mean_iouC})
ppre=gResMCSegPre(patchsize=224,bordersize=16,overprop=0.3)
img, mask = dload()
imgres = ppre.preAImgMulti(img, model, 9)
mskImg = onehot_to_rgb(imgres, color_dict)
fpath = "/tmp/zurich1img_pre.jpg"
cv2.imwrite(fpath, cv2.cvtColor(mskImg, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
y_pred = imgres.flatten()
y_true = mask.flatten()
iou = compute_iou(imgres, mask)
jacard = jaccard(imgres, mask)
print("iou : " + str(iou) + '; jacard is ', jacard)

```
## Collaboration

Welcome to contact Dr. Lianfa Li (Email: lspatial@gmail.com or lilf@lreis.ac.cn). 