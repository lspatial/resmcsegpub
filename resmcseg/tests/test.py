import os
import cv2
from keras.models import load_model,model_from_json
from resmcseg.util.segmetrics import compute_iou,jaccard,mean_iouC,miou,mean_iou
from resmcseg.model.gresmcseg_pre import gResMCSegPre
from resmcseg.model.resizelayer import ResizeLayer
from resmcseg.model.pretrainedmodel import downloadPretrainedModel
from resmcseg.util.helper import bce_dice_loss,jaccard_coef,jaccard_coef_int,jaccard_coef1
from resmcseg.util.helper import onehot_to_rgb,color_dict
from resmcseg.data import data

def getResImg(resimg):
    return onehot_to_rgb(resimg, color_dict)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str('0')

print('Starting to download the model... ...')
modelFl='/tmp/model_strwei.h5'
downloadPretrainedModel('ZURICH',destination=modelFl)
model = load_model(modelFl,custom_objects={'ResizeLayer': ResizeLayer,'bce_dice_loss':bce_dice_loss,
               'mean_iou':mean_iou,'jaccard_coef':jaccard_coef, 'jaccard_coef1':jaccard_coef1,'miou':miou,
                                                   'jaccard_coef_int':jaccard_coef_int,'mean_iouC': mean_iouC})
print('Downloading of the model Done!')
ppre=gResMCSegPre(patchsize=224,bordersize=16,overprop=0.3)
img, mask = data()
imgres = ppre.preAImgMulti(img, model, 9)
mskImg = getResImg(imgres)
fpath = "/tmp/zurich1img_pre.jpg"
cv2.imwrite(fpath, cv2.cvtColor(mskImg, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
y_pred = imgres.flatten()
y_true = mask.flatten()
iou = compute_iou(imgres, mask)
jacard = jaccard(imgres, mask)
print("iou : " + str(iou) + '; jacard is ', jacard)
