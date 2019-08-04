# -*- coding: utf-8 -*-
"""
Major class: gAutoUnetClass

This class provides the functionality of deep residual multiscale
segmenter for remotely sensed images based on the KERAS environment.
The users need to set up their arguments of the model for construction
of the deep FCN. These parameters include number of layers, number of
nodes for each decoding layer, type of residuals, initializer, dropout
 rate, flag for batch normalization, regularizer, and multiscale type
 to obtain a residual network.

Author: Lianfa Li
Date: 2019-07-29

"""


from keras.models import Model
from keras.layers import Input, Lambda, concatenate, Concatenate,Conv2D, MaxPooling2D,RepeatVector,Reshape,add
from keras.layers import UpSampling2D, BatchNormalization, Dropout, Cropping2D, ZeroPadding2D,Conv2DTranspose
import math
from keras.layers.advanced_activations import ELU,ReLU
from .resizelayer import ResizeLayer
import keras.backend  as K
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf

class gResMCSeg:
    """
      Formal class of Deep Residual Multiscale Segmenter
      This class provides flexible architecture for deep extensive residual multiscale segmenter

     # Examples

     ```python
         # Set the dict type for multiple key-value paramaters :

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
         input_shape: shape of the input images including width, height and channel etc.;
         start_filter: number of filters for the starting layer ;
         depth: Number of the layers ;
         kwargs: other key-values parameters including
              'inc_step': incremental step for the number of filters for each decoding layer, default: 2,
               'up2deconv': use of upsampling rather than deconvolutions, default: True,
               'activation': activation function for the hidden layer, default: 'relu',
               'nfeature': number of features for each image sample, default: 0,
               'dropout': dropout rate, default: 0.5,
               'batchnorm': use of batch normalization, default: True,
               'cropsize': use of crop size for the input image, default: None,
               'ismaxpool': use of maximum pooling, default: True,
               'isconvcon': use of concatenation rather than residual connection for shortcut, default:  False,
               'k_initializer': initializer for the parameters, default: 'glorot_uniform',
               'residual': use of the residual unit, default: True,
               'residualAuto': use of nested residual connections, default: True,
               'multsctype': type of multiscale, four options:{resize, dilated (ASPP), both, None}, default:None,
               'mctotallevels': number of multiscle layers, default: 10,
               'mcbothcoder': False,
               'mcfilters': number of multiscale filter, default: None, setup by filters of each convolutions
      #Output:
         The model with the input and output tensor
    """

    def __init__(self,input_shape,start_filter,depth,**kwargs):
        self.input_shape=input_shape
        self.start_filter=start_filter
        self.depth=depth
        self.inc_step=kwargs['inc_step'] if 'inc_step' in  kwargs else 2
        self.up2deconv=kwargs['up2deconv'] if 'up2deconv' in  kwargs else True
        self.activation=kwargs['activation'] if 'activation' in  kwargs else 'relu'
        self.nfeature=kwargs['nfeature'] if 'nfeature' in  kwargs else 0
        self.dropout=kwargs['dropout'] if 'dropout' in  kwargs else 0.5
        self.batchnorm=kwargs['batchnorm'] if 'batchnorm' in  kwargs else True
        self.cropsize=kwargs['cropsize'] if 'cropsize' in  kwargs else None
        self.ismaxpool=kwargs['ismaxpool'] if 'ismaxpool' in  kwargs else True
        self.isconvcon=kwargs['isconvcon'] if 'isconvcon' in  kwargs else False
        self.k_initializer= kwargs['k_initializer'] if 'k_initializer' in  kwargs else 'glorot_uniform'
        self.residual=kwargs['residual'] if 'residual' in  kwargs else False
        self.residualAutoType=kwargs['residualAutoType'] if 'residualAutoType' in  kwargs else 0
        self.multsctype=kwargs['multsctype'] if 'multsctype' in  kwargs else None
        self.mctotallevels=kwargs['mctotallevels'] if 'mctotallevels' in  kwargs else 10
        self.mcbothcoder=kwargs['mcbothcoder'] if 'mcbothcoder' in  kwargs else False
        self.mcfilters=kwargs['mcfilters'] if 'mcfilters' in  kwargs else None

        self.nmaxpool =-1
        self.mclevel = 1


    def conv2dBlock(self, inlayer, nfilter, dropout=0):
        """
          Function to obtain one decoding or encoding building unit (consisting of multiple convolutions)
          :param inlayer: input layer ;
          :param nfilter: number of filters ;
          :param dropout: droput rate, default: 0 (no dropout layer applied)
          :return: the output layer of the encoding or decoding layer
        """
        if self.residual:
            #shortcut = Conv2D(nfilter, 3, activation=self.activation, padding='same',
            #                      kernel_initializer=self.k_initializer)(inlayer)
            shortcut=inlayer
            #shortcut = BatchNormalization()(shortcut) if self.batchnorm else shortcut
        layer = Conv2D(nfilter, 3, activation=self.activation, padding='same',kernel_initializer=self.k_initializer)(shortcut if self.residual else inlayer)
        layer = BatchNormalization()(layer) if self.batchnorm else layer
        if self.residual:
            shortcut=layer
        layer = Dropout(dropout)(layer) if dropout else layer
        layer = Conv2D(nfilter, 3, activation=self.activation, padding='same',kernel_initializer=self.k_initializer)(layer)
        layer = BatchNormalization()(layer) if self.batchnorm else layer
        if self.isconvcon :
             return Concatenate()([inlayer, layer])
 #            return add([inlayer, layer])
        if self.residual:
            layer=add([shortcut,layer])
            #layer=ReLU()(layer) if self.activation == 'relu' else ELU()(layer)
            return layer
        return layer

    def levelsBlock(self, inlayer, nfilter, depth, dropout,input_fea_layer=None):
        """
          Function to recursively construct the autoencoder architecture (encoding-coding-decoding)
          :param inlayer: input layer ;
          :param nfilter: number of filters ;
          :param depth: number of decoding units ;
          :param dropout: droput rate, default: 0 (no dropout layer applied)
          :param input_fea_layer: input of a feature, default: None  ;
          :return: the output tensor of this autoencoder
        """
        if depth > 0:
            layer = self.conv2dBlock(inlayer, nfilter)
            if self.ismaxpool:
                mlayer = MaxPooling2D()(layer)
                self.nmaxpool+=1
                if self.multsctype is not None:
                    mlayer=self.embeddedMcLayer(mlayer)
            else:
                mlayer = Conv2D(nfilter, 3, strides=2, padding='same',kernel_initializer=self.k_initializer)(layer)
            mlayer = self.levelsBlock(mlayer, int(self.inc_step * nfilter),depth - 1, dropout,input_fea_layer)
            if self.up2deconv:
                mlayer = UpSampling2D()(mlayer)
                if self.mcbothcoder and self.multsctype is not None:
                    mlayer = self.embeddedMcLayer(mlayer)
                mlayer = Conv2D(nfilter, 2, activation=self.activation, padding='same',kernel_initializer=self.k_initializer)(mlayer)
            else:
                mlayer = Conv2DTranspose(nfilter, 3, strides=2, activation=self.activation,
                                         padding='same',kernel_initializer=self.k_initializer)(mlayer)
            if self.residualAutoType==1:
                clayer = add([layer, mlayer])
            elif self.residualAutoType==0:
                clayer = Concatenate()([layer, mlayer])
            elif self.residualAutoType==2:
                clayer = add([layer, mlayer])
                clayer = Concatenate()([clayer, mlayer])
            elif self.residualAutoType==3:
                clayer = add([layer, mlayer])
                clayer = Concatenate()([clayer, layer])
            mlayer = self.conv2dBlock(clayer, nfilter)
        else:
            if self.nfeature==0 or self.nmaxpool<1:
                mlayer = self.conv2dBlock(inlayer, nfilter, dropout)
            else:
                krow = int(self.input_shape[0]/math.pow(2, self.nmaxpool))
                kcol = int(self.input_shape[1]/math.pow(2, self.nmaxpool))
                f_repeat = RepeatVector(krow *kcol)(input_fea_layer)
                f_conv = Reshape((krow, kcol, self.nfeature))(f_repeat)
                flayer = concatenate([inlayer, f_conv], -1)
                mlayer = self.conv2dBlock(flayer, nfilter, dropout)
        return mlayer

    def embeddedMcLayer(self,inlayer):
        """
          Function to multiscale input
          :param inlayer: input layer ;
          :return: this multiscale layer's output to be linked to its corresponding decoding layer
        """
        tshp = K.int_shape(inlayer)
        nfilter= int(tshp[3] / 4)
        if self.mcfilters is not None:
            nfilter=self.mcfilters[self.mclevel-1]
        if self.mclevel <= self.mctotallevels:
            if self.multsctype=='resize':
                mclayer = self.MultiScalePreLayer(nfilter=nfilter, output_dim=(tshp[1], tshp[2]))
                mlayer = Concatenate()([inlayer, mclayer])
                self.mclevel = self.mclevel + 1
            elif self.multsctype=='dilated':
                mclayer = self.MultiScaleDilateLayer(nfilter=nfilter, output_dim=(tshp[1], tshp[2]))
                mlayer = Concatenate()([inlayer, mclayer])
                self.mclevel = self.mclevel + 1
            elif self.multsctype=='both':
                mclayer = self.MultiScalePreLayer(nfilter=nfilter, output_dim=(tshp[1], tshp[2]))
                dilatedlayer = self.MultiScaleDilateLayer(nfilter=nfilter, output_dim=(tshp[1], tshp[2]))
                mlayer = Concatenate()([inlayer, dilatedlayer,mclayer])
                self.mclevel = self.mclevel + 1
            else:
                mlayer=inlayer
        return mlayer

    def MultiScalePreLayer(self, nfilter, output_dim):
        """
        Function to resizing multiscale layer
        :param nfilter: number of filters ;
         :param output_dim: dimension of the output ;
        :return: this multiscale layer's output
        """
        rzlayer = ResizeLayer(output_dim=output_dim)(self.inputlayer)
        outlayer = self.conv2dBlock(rzlayer, nfilter)
        return outlayer

    def MultiScaleDilateLayer(self, nfilter, output_dim):
        """
        Function to ASPP multiscale layer
        :param nfilter: number of filters ;
         :param output_dim: dimension of the output ;
        :return: this multiscale layer's output
        """
        layer = self.inputlayer
        dilations = [1, 4, 8, 16]
        aspp1 = Conv2D(nfilter, 3, activation='relu', padding='same', kernel_initializer=self.k_initializer,
                       dilation_rate=dilations[0])(layer)
        aspp1 = BatchNormalization()(aspp1)
        aspp2 = Conv2D(nfilter, 3, activation='relu', padding='same', kernel_initializer=self.k_initializer,
                       dilation_rate=dilations[1])(layer)
        aspp2 = BatchNormalization()(aspp2)
        aspp3 = Conv2D(nfilter, 3, activation='relu', padding='same', kernel_initializer=self.k_initializer,
                       dilation_rate=dilations[2])(layer)
        aspp3 = BatchNormalization()(aspp3)
        aspp4 = Conv2D(nfilter, 3, activation='relu', padding='same', kernel_initializer=self.k_initializer,
                       dilation_rate=dilations[3])(self.inputlayer)
        aspp4 = BatchNormalization()(aspp4)
        outlayer = Concatenate()([aspp1, aspp2, aspp3,aspp4])
        #outlayer = add([aspp1, aspp2, aspp3, aspp4])
        outlayer = BatchNormalization()(outlayer)
        outlayer = ReLU()(outlayer)
        outlayer = ResizeLayer(output_dim=output_dim)(outlayer)
        return outlayer


    def cropLayer(self,inlayer):
        """
        Function to cropping an input layer
        :param inlayer: input layer;
        :return: this output layer
        """
        if self.cropsize != None:
            croplayer = Cropping2D(cropping=((self.cropsize,self.cropsize),(self.cropsize,self.cropsize)))(inlayer)
            clayer = BatchNormalization(axis=3)(croplayer) if self.batchnorm else croplayer
            outlayer = ReLU()(clayer) if self.activation == 'relu' else ELU()(clayer)
            return outlayer
        return inlayer

    def ResMCNet(self,outfilter=1,ngpu=1):
        """
        Function to cropping an input layer
        :param outfilter: number of output units (1: for binary output; n for multiclass output);
        :param ngpu: number of GPU, default 1;
        :return: the model constructed with the input and output layer
        """
        inputlayer = Input(shape=self.input_shape,name='img')
        self.inputlayer = inputlayer
        self.nmaxpool=0
        if self.nfeature==0:
            outlayer = self.levelsBlock(inputlayer, self.start_filter, self.depth, self.dropout)
            outlayer = self.cropLayer(outlayer)
            outlayer = Conv2D(outfilter, 1, activation='sigmoid')(outlayer)
            return Model(inputs=inputlayer, outputs=outlayer)
        input_fea_layer = Input((self.nfeature,), name='feat')
        outlayer = self.levelsBlock(inputlayer, self.start_filter, self.depth, self.dropout,input_fea_layer)
        outlayer=self.cropLayer(outlayer)
        outlayer = Conv2D(outfilter, 1, activation='sigmoid',kernel_initializer=self.k_initializer)(outlayer)
        if ngpu>1:
            with tf.device("/cpu:0"):
                fmodel = Model(inputs=[inputlayer, input_fea_layer], outputs=outlayer)
            fmodel = multi_gpu_model(fmodel, gpus=ngpu)
            return fmodel
        return Model(inputs=[inputlayer, input_fea_layer], outputs=outlayer)

