import math
import cv2
import numpy as np
import sys


class gResMCSegPre:
    """
          Formal class using Deep Residual Multiscale Segmenter for prediction

         # Examples

         ```python
             #

             dkwargs={'inc_step': 2, 'up2deconv': True, 'activation': 'relu', 'nfeature': 0, 'dropout': 0.5, 'batchnorm': True,
                'cropsize': None, 'ismaxpool': True, 'isconvcon': False, 'k_initializer': 'glorot_uniform',
                'residual': False, 'residualAutoType': 0, 'multsctype': None, 'mctotallevels': 10,
                'mcbothcoder': False, 'mcfilters': [16, 16,16,16,16,16,16,16,16,16,16,16,16,16],'cropsize':16}
             start_filter=32
             modelCls = gResMCSeg(input_shape=(256, 256,20), start_filter= start_filter, depth=depth, **dkwargs)
             model = modelCls.ResMCNet()
             model.summary()
             model.compile(optimizer="adam", loss= K_nJIBCE_loss,
                   metrics=['accuracy',K_nJI, K_nJIBCE_loss])
        ```
          # Arguments
             patchsize: Input size of a patch sample ;
             bordersize: Border size to filter out border effect  ;
             overprop: Overlapping proportion for two neighbor images, default: 0, no overlapping ;

          #Output:
             Predicted mask matrix (np array type)
             Predicted mask matrix (np array type)
        """
    def __init__(self,patchsize,bordersize,overprop=0):
        self.patchsize=patchsize
        self.border_size=bordersize
        self.overprop=overprop

    def preAImgBi(self,image, model):

        """
          Function to predict binary mask based on the trained model
          :param image: input image ;
          :param model: the trained model ;
          :return: the matrix of predicted mask (0,1)
        """
        imarray = np.array(image)
        imarray = cv2.copyMakeBorder(imarray, self.border_size, self.border_size, self.border_size,
                                     self.border_size, cv2.BORDER_REFLECT)
        nrow, ncol, _ = imarray.shape
        if (nrow < ncol and nrow < self.patchsize) or (nrow > ncol and ncol < self.patchsize):
            return None
        imgres = np.zeros((image.shape[0], image.shape[1], 2))
        # img=imarray/255.0
        stride = math.floor((1 - self.overprop) * self.patchsize)
        for ir in range(0, nrow + 1, stride):
            for jc in range(0, ncol + 1, stride):
                ifrm = ir
                jfrm = jc
                it = ifrm + self.patchsize + self.border_size * 2
                jt = jfrm + self.patchsize + self.border_size * 2
                if it > nrow:
                    it = nrow
                    ifrm = nrow - self.patchsize - self.border_size * 2
                if jt > ncol:
                    jt = ncol
                    jfrm = ncol - self.patchsize - self.border_size * 2
                #                    it = it + self.border_size * 2
                #                    jt = jt + self.border_size * 2
                imgB = imarray[ifrm:it, jfrm:jt, :]
                imgB = imgB[None, :, :, :]
                # print(ir,jc,jfrm,jt,imgB.shape)
                res = model.predict(imgB)
                i2c = (ifrm, ifrm + self.patchsize)
                j2c = (jfrm, jfrm + self.patchsize)
                ti2 = (self.border_size, self.patchsize + self.border_size)
                tj2 = (self.border_size, self.patchsize + self.border_size)
                #        print(np.max(res), np.min(res))
                imgres[i2c[0]:i2c[1], j2c[0]:j2c[1], 0] = imgres[i2c[0]:i2c[1], j2c[0]:j2c[1], 0] \
                                                          + res[0, ti2[0]:ti2[1], tj2[0]:tj2[1], 0]
                imgres[i2c[0]:i2c[1], j2c[0]:j2c[1], 1] = imgres[i2c[0]:i2c[1], j2c[0]:j2c[1], 1] + 1.0
        imgres = imgres[:, :, 0] / imgres[:, :, 1]
        print("total res:", np.max(imgres), np.min(imgres))
        return imgres


    def preAImgMulti(self, image,model,ncls=9):
        """
          Function to predict multiclass mask based on the trained model
            :param image: input image ;
            :param model: the trained model ;
            :param ncls: number of classes ;
            :return: the matrix of predicted mask, ncls dimensions
        """
        imarray = np.array(image)
        imarray = cv2.copyMakeBorder(imarray, self.border_size, self.border_size, self.border_size,
                                 self.border_size, cv2.BORDER_REFLECT)
        nrow, ncol, _ = imarray.shape
        if (nrow < ncol and nrow < self.patchsize) or (nrow > ncol and ncol < self.patchsize):
            return
        imgres = np.zeros((image.shape[0], image.shape[1], ncls + 1))
        stride = math.floor((1 - self.overprop) * self.patchsize)
        for ir in range(0, nrow + 1, stride):
            for jc in range(0, ncol + 1, stride):
                ifrm = ir
                jfrm = jc
                it = ifrm + self.patchsize + self.border_size * 2
                jt = jfrm + self.patchsize + self.border_size * 2
                if it > nrow:
                    it = nrow
                    ifrm = nrow - self.patchsize - self.border_size * 2
                if jt > ncol:
                    jt = ncol
                    jfrm = ncol - self.patchsize - self.border_size * 2
                imgB = imarray[ifrm:it, jfrm:jt, :]
                imgB = imgB[None, :, :, :]
                # print(ir,jc,jfrm,jt,imgB.shape)
                res = model.predict(imgB)
                i2c = (ifrm, ifrm + self.patchsize)
                j2c = (jfrm, jfrm + self.patchsize)
                imgres[i2c[0]:i2c[1], j2c[0]:j2c[1], 0:ncls] = imgres[i2c[0]:i2c[1], j2c[0]:j2c[1], 0:ncls] + res
                imgres[i2c[0]:i2c[1], j2c[0]:j2c[1], ncls] = imgres[i2c[0]:i2c[1], j2c[0]:j2c[1], ncls] + 1.0
        imgres = imgres[:, :, 0:ncls] / imgres[:, :, ncls][:, :, None]
        print("total res:", np.max(imgres), np.min(imgres))
        pindex = np.argmax(imgres, axis=2)
        imgres = np.zeros(imgres.shape)
        for c in range(imgres.shape[0]):
            for r in range(imgres.shape[1]):
                imgres[c, r, pindex[c, r]] = 1
        return imgres


