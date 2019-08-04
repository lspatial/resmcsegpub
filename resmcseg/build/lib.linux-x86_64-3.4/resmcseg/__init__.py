# -*- coding: utf-8 -*-
""" Library of Autoencoder-based Residual Deep Network

The python library of Deep Residual Multiscale Segmenter (autonet).
Current version just supports the KERAS package of deep learning and
will extend to the others in the future.

Major modules
        model:gResMCSeg: major class to obtain a deep extensive residual multiscale
                  FCN. You can setup its aruments. See the class and its
                  member functions' help for details.
              gResMCSegPre: major class to make semantic segmentation for binary and multi class.
              pretrainedmodel: function to download the pretrained models using the DSTL and
                            ZURICH datasets from the Google cloud
        util: segmetrics: main metrics including jaccard index, MIoU, and loss functions etc.
              helper: helper functions including color mapping etc.
        data  function to access one image for Zurich to test the model's prediction.

Github source: https://github.com/lspatial/
Author: Lianfa Li
Date: 2018-10-01

"""

#import pkgutil
#__path__ = pkgutil.extend_path(__path__, __name__)

from resmcseg import model
from resmcseg import data
from resmcseg import util


__version__ = '0.1.3'

