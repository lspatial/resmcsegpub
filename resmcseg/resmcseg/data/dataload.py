import pkg_resources
import tifffile as tiff
from osgeo import gdal
import numpy as np

from resmcseg.util.helper import MAXPIXELVAL,rgb_to_onehot,color_dict

def aimgload(name='zurich'):
    """
    Function to obtain the sample data.
    :param name: string, the name for each of two datasets.
          'zurich': One image from the Zurich dataset
          'dstl': not available now due to too large data
    :return: img and mask
    """
    if name!='zurich':
        print("Not supported for other data")
        return (None,None)
    fl=pkg_resources.resource_filename(__name__, '/zh17.tif')
    img = tiff.imread(fl)
    img = img.astype(np.float32) / MAXPIXELVAL
    fl = pkg_resources.resource_filename(__name__, '/zh17_gt.tif')
    gt = gdal.Open(fl).ReadAsArray()
    gt = np.rollaxis(gt, 0, 3)
    gt_onehot = rgb_to_onehot(gt, color_dict)
    print(gt_onehot.shape)
    return (img, gt_onehot)
