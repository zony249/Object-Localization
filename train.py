#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

import tensorflow as tf
import tensorboard
from dataloader import *


# ## Indexing our Dataset

# In[2]:


BBOX_TRAIN_PATH = "../ImageNet/BBOX_train/"
IMG_TRAIN_PATH = "../ImageNet/IMG_train/"

BBOX_VAL_PATH = "../ImageNet/BBOX_val/"
IMG_VAL_PATH = "../ImageNet/IMG_val/"


data_index = Data_index(bb_train_path=BBOX_TRAIN_PATH, 
                        img_train_path=IMG_TRAIN_PATH, 
                        bb_val_path=BBOX_VAL_PATH, 
                        img_val_path=IMG_VAL_PATH)
data_index.populate()


num_training_data = data_index.trainsize()
num_validation_data = data_index.valsize()
num_classes = data_index.num_classes

print(f"\n\nnumber of training data: {num_training_data}")
print(f"number of validation data: {num_validation_data}")
print(f"number of classes: {num_classes}")


# In[ ]:





# ## Defining our Model

# In[3]:


from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
tf.config.experimental.enable_tensor_float_32_execution(enabled=True)



gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


# In[4]:


from tensorflow.keras import Model
from tensorflow.keras.layers import SeparableConv2D, Conv2D, DepthwiseConv2D, MaxPool2D, Flatten, Dense, Activation, BatchNormalization, Softmax
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar

class ImageNet(Model):
    def __init__(self, rows, cols, channels, num_of_classes=1000, reg=1e-30, drop=[]):
        super().__init__()
        
        # in: (416, 416, 3)
        self.norm = BatchNormalization(axis=-1, input_shape=(rows, cols, channels))
        self.conv1_1 = SeparableConv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding="same", depthwise_initializer="he_normal", pointwise_initializer= "he_normal")
        self.a1_1 = Activation("relu")
        self.conv1_2 = SeparableConv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding="same", depthwise_initializer="he_normal", pointwise_initializer= "he_normal")
        self.a1_2 = Activation("relu")
        self.conv1_3 = SeparableConv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding="same", depthwise_initializer="he_normal", pointwise_initializer= "he_normal")
        self.bn1 = BatchNormalization(-1)
        self.a1_3 = Activation("relu")
        self.maxpool1 = MaxPool2D(pool_size=(4, 4))
        # out: (104, 104, 32)
        
        # in: (104, 104, 32)
        self.conv2_1 = SeparableConv2D(filters=192, kernel_size=(5, 5), strides=(1, 1), padding="same", depthwise_initializer="he_normal", pointwise_initializer= "he_normal")
        self.a2_1 = Activation("relu")
        self.conv2_2 = SeparableConv2D(filters=192, kernel_size=(5, 5), strides=(1, 1), padding="same", depthwise_initializer="he_normal", pointwise_initializer= "he_normal")
        self.a2_2 = Activation("relu")
        self.conv2_3 = SeparableConv2D(filters=192, kernel_size=(5, 5), strides=(1, 1), padding="same", depthwise_initializer="he_normal", pointwise_initializer= "he_normal")
        self.bn2 = BatchNormalization(-1)
        self.a2_3 = Activation("relu")
        self.maxpool2 = MaxPool2D(pool_size=(5, 5), strides=(1, 1))
        # out: (100, 100, 192)
        
        # in: (100, 100, 192)
        self.incept3_1 = Inception(filters1=128, filters3=256, filters5=256, filters_p=128) # filters: 128, 256, 256, 128
        self.a3_1 = Activation("relu")
        self.incept3_2 = Inception(filters1=128, filters3=256, filters5=256, filters_p=128) # filters: 128, 256, 256, 128
        self.a3_2 = Activation("relu")
        self.incept3_3 = Inception(filters1=128, filters3=256, filters5=256, filters_p=128) # filters: 128, 256, 256, 128
        self.bn3 = BatchNormalization(-1)
        self.a3_3 = Activation("relu")
        self.maxpool3 = MaxPool2D(pool_size=(2, 2))
        # out: (50, 50, 768)
        
        # in: (50, 50, 768)
        self.incept4_1 = Inception(filters1=256, filters3=512, filters5=512, filters_p=256) # filters: 256, 512, 512, 256
        self.a4_1 = Activation("relu")
        self.incept4_2 = Inception(filters1=256, filters3=512, filters5=512, filters_p=256) # filters: 256, 512, 512, 256
        self.a4_2 = Activation("relu")
        self.incept4_3 = Inception(filters1=256, filters3=512, filters5=512, filters_p=256) # filters: 256, 512, 512, 256
        self.bn4 = BatchNormalization(-1)
        self.a4_3 = Activation("relu")
        self.maxpool4 = MaxPool2D(pool_size=(4, 4), strides=(2, 2))
        # out: (24, 24, 1536)
        
        # in: (24, 24, 1536)
        self.incept5_1 = Inception(filters1=512, filters3=512, filters5=512, filters_p=512) # filters: 512, 768, 512, 512
        self.a5_1 = Activation("relu")
        self.incept5_2 = Inception(filters1=512, filters3=512, filters5=512, filters_p=512) # filters: 512, 768, 512, 512
        self.a5_2 = Activation("relu")
        self.incept5_3 = Inception(filters1=512, filters3=512, filters5=512, filters_p=512)# filters: 512, 768, 512, 512
        self.bn5 = BatchNormalization(-1)
        self.a5_3 = Activation("relu")
        self.maxpool5 = MaxPool2D(pool_size=(2, 2))
        # out: (12, 12, 2048)
        
        # in: (12, 12, 2048)
        self.conv_existance = SeparableConv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding="same", depthwise_initializer="glorot_normal", pointwise_initializer= "glorot_normal")
        self.a_existance = Activation("sigmoid", name="exist")
        
        self.conv_bbox = SeparableConv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding="same", depthwise_initializer="he_normal", pointwise_initializer= "he_normal")
        self.a_bbox = Activation("linear", name="bbox")
        
        self.conv_class = SeparableConv2D(filters=num_of_classes, kernel_size=(1, 1), strides=(1, 1), padding="same", depthwise_initializer="glorot_normal", pointwise_initializer= "glorot_normal")
        self.a_class = Softmax(axis=-1, name="classes")
        
        
        # other stuff
        self.reg = reg
        
        
       
    def call(self, inputs, training=False):
        
        # Block 1
        #x = tf.cast(inputs, tf.float16)
        x = self.norm(inputs, training=training)    # |
        x = self.conv1_1(x)                         # |
        x_res = self.a1_1(x)                        # |----┐
                                                    # |    |
        x = self.conv1_2(x_res)                     # |    |
        x = self.a1_2(x)                            # |    |
                                                    # |    |
        x = self.conv1_3(x)                         # |    |
        x = self.bn1(x + x_res, training=training)  # |<---┘
        x = self.a1_3(x)                            # |
        x = self.maxpool1(x)                        # |
                                                    # |
        # Block 2                                   # |
        x = self.conv2_1(x)                         # |
        x_res = self.a2_1(x)                        # |----┐
                                                    # |    |
        x = self.conv2_2(x_res)                     # |    |
        x = self.a2_2(x)                            # |    |
                                                    # |    |
        x = self.conv2_3(x)                         # |    |
        x = self.bn2(x + x_res, training=training)  # |<---┘
        x = self.a2_3(x)                            # |
        x = self.maxpool2(x)                        # |    
                                                    # |
        # Block 3                                   # |
        x = self.incept3_1(x)                       # |
        x_res = self.a3_1(x)                        # |----┐
                                                    # |    |
        x = self.incept3_2(x_res)                   # |    |
        x = self.a3_2(x)                            # |    |
                                                    # |    |
        x = self.incept3_3(x)                       # |    |
        x = self.bn3(x + x_res, training=training)  # |<---┘
        x = self.a3_3(x)                            # |
        x = self.maxpool3(x)                        # |     
                                                    # |
        # Block 4                                   # |
        x = self.incept4_1(x)                       # |
        x_res = self.a4_1(x)                        # |----┐
                                                    # |    |
        x = self.incept4_2(x_res)                   # |    |
        x = self.a4_2(x)                            # |    |
                                                    # |    |
        x = self.incept4_3(x)                       # |    |
        x = self.bn4(x + x_res, training=training)  # |<---┘
        x = self.a4_3(x)                            # |
        x = self.maxpool4(x)                        # |      
                                                    # |
        # Block 5                                   # |
        x = self.incept5_1(x)                       # |
        x_res = self.a5_1(x)                        # |----┐
                                                    # |    |
        x = self.incept5_2(x_res)                   # |    |
        x = self.a5_2(x)                            # |    |
                                                    # |    |
        x = self.incept5_3(x)                       # |    |
        x = self.bn5(x + x_res, training=training)  # |<---┘
        x = self.a5_3(x)                            # |
        x = self.maxpool5(x)                        # |
                                                    # |
        # Output Block                          ##### |
        x_exist = self.conv_existance(x)        # ┌<--┤
        a_exist = self.a_existance(x_exist)     # |   |
                                                # |   |
        x_bbox = self.conv_bbox(x)              # |<--┤
        a_bbox = self.a_bbox(x_bbox)            # |   |
                                                # |   |
        x_classes = self.conv_class(x)          # |<--┘
        a_classes = self.a_class(x_classes)     # |
                                                # ˅
        
        # Final Concat
        xout = tf.concat([a_exist, a_bbox, a_classes], axis=-1)
#        print(f"xout shape: {tf.shape(xout)}")
        #print(tf.reduce_sum(self.conv1_1.get_weights()[1]) )
        return xout
        
    

   
    def fit(self, t_gen, loss_fn, opt, epochs=1, steps_per_epoch=1, v_gen=None, val_steps=1, metrics=[]):
        
        for i in range(epochs):
            
            print(f"Epoch {i+1}/{epochs}")
            
            metric_names = [x.__name__ for x in metrics]
            metric_names.insert(0, "loss")
            progbar = Progbar(steps_per_epoch, stateful_metrics=metric_names)
            
            
            for j in range(steps_per_epoch):
                X_train, Y_train = next(t_gen)
#                 with tf.GradientTape() as tape:
#                     Y_pred = self(X_train, training=True)
#                     loss = loss_fn(Y_train, Y_pred)
#                 grads = tape.gradient(loss, self.trainable_weights)
#                 opt.apply_gradients(zip(grads, self.trainable_weights))
                loss, Y_pred = self.train_step(X_train, Y_train, opt, loss_fn, metrics)
                #print(f"\rEpoch: {i+1}/{epochs}, iter: {j + 1}/{steps_per_epoch} -- loss: {loss:.4f}", end="")
                #for k in range(len(mtx)):
                   #print(f" -- {metrics[k].__name__}: {mtx[k]:.4f}", end="")
                mtx = [("loss", loss)]
                for met in metrics:
                    mtx.append((met.__name__ , met(Y_train, Y_pred)))
                progbar.add(1, values=mtx)
            print(" ")
            
    @tf.function(jit_compile=True)
    def train_step(self, x, y, opt, loss_fn, metrics=[]):
      
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = loss_fn(y, y_pred)
        grads = tape.gradient(loss, self.trainable_weights)
        opt.apply_gradients(zip(grads, self.trainable_weights))
        
#         mtx = []
#         for met in metrics:
#             mtx.append((met.__name__ , met(y, y_pred)))
        
        return loss, y_pred
    
class Inception(tf.keras.layers.Layer):
    def __init__(self, filters1, filters3, filters5, filters_p):
        super().__init__()
        self.conv1 = Conv2D(filters=filters1, kernel_size=(1, 1), strides=(1, 1), padding="same", kernel_initializer="he_normal")
        
        self.conv_pre3 = Conv2D(filters=filters3, kernel_size=(1, 1), strides=(1, 1), padding="same", kernel_initializer="he_normal")
        self.conv3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding="same", depthwise_initializer="he_normal")
        
        self.conv_pre5 = Conv2D(filters=filters5, kernel_size=(1, 1), strides=(1, 1), padding="same", kernel_initializer="he_normal")
        self.conv5 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding="same", depthwise_initializer="he_normal")
        
        self.mpool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same")
        self.poolconv = Conv2D(filters=filters_p, kernel_size=(1, 1), strides=(1, 1), padding="same", kernel_initializer= "he_normal")
        
    def call(self, input_tensor, training=True):
        
        x1 = self.conv1(input_tensor)
        
        x3 = self.conv_pre3(input_tensor)
        x3 = self.conv3(x3)
        
        x5 = self.conv_pre5(input_tensor)
        x5 = self.conv5(x5)
        
        xp = self.mpool(input_tensor)
        xp = self.poolconv(xp)
        
        xout = tf.concat([x1, x3, x5, xp], axis=-1)
        return xout


# In[5]:


from tensorflow.keras.losses import Loss

class CostFunction(Loss):
    def __init__(self):
        super().__init__()
    
    def call(self, y_true, y_pred):
        
        b = y_true.shape[0]
        h = y_true.shape[1]
        w = y_true.shape[2]
        c = y_true.shape[3]-5
        
        
        # sum(-ylog(y')-(1-y)log(1-y')) / batch_size
        exist_true = tf.cast(y_true[:, :, :, 0], tf.float32)
        exist_pred = tf.cast(y_pred[:, :, :, 0] * (1-1e-5) + (1e-5)/2, tf.float32)
        exist_loss = tf.reduce_sum(-exist_true * tf.math.log(exist_pred)-(1-exist_true)*tf.math.log(1-exist_pred), axis=None)/b/w/h

        # sum(sum((y-y')^2, axis=-1) * exist)/ batch_size
        bbox_true = tf.cast(y_true[:, :, :, 1:5], tf.float32) * tf.expand_dims(exist_true, axis=-1)
        bbox_pred = tf.cast(y_pred[:, :, :, 1:5], tf.float32) * tf.expand_dims(exist_true, axis=-1)
        bbox_loss = tf.reduce_sum(tf.pow(bbox_true - bbox_pred, 2), axis=None)/b
        
        # sum(sum(-ylog(y'), axis=-1) * exist, axis=None)/ batch_size
        class_true = tf.cast(y_true[:, :, :, 5:], tf.float32)
        class_pred = tf.cast(y_pred[:, :, :, 5:] * (1-1e-5) + (1e-5)/2, tf.float32)
        class_loss = tf.reduce_sum(tf.reduce_sum(-class_true*tf.math.log(class_pred), axis=-1) * exist_true, axis=None)/b
        
        total_loss = 100*exist_loss + 70*bbox_loss + 10*class_loss
        return total_loss
        
def existance(y_true, y_pred):
    
    b = y_true.shape[0]
    h = y_true.shape[1]
    w = y_true.shape[2]
    c = y_true.shape[3]-5
    # sum(-ylog(y')-(1-y)log(1-y')) / batch_size
    exist_true = tf.cast(y_true[:, :, :, 0], tf.float32)
    exist_pred = tf.cast(y_pred[:, :, :, 0] * (1-1e-5) + (1e-5)/2, tf.float32)  
    exist_loss = tf.reduce_sum(-exist_true * tf.math.log(exist_pred)-(1-exist_true)*tf.math.log(1-exist_pred), axis=None)/b/w/h
    return 100*exist_loss
    
def bbox(y_true, y_pred):
    b = y_true.shape[0]
    h = y_true.shape[1]
    w = y_true.shape[2]
    c = y_true.shape[3]-5
    # sum(sum((y-y')^2, axis=-1) * exist)/ batch_size
    exist_true = tf.cast(y_true[:, :, :, 0], tf.float32)
    bbox_true = tf.cast(y_true[:, :, :, 1:5], tf.float32) * tf.expand_dims(exist_true, axis=-1)
    bbox_pred = tf.cast(y_pred[:, :, :, 1:5], tf.float32) * tf.expand_dims(exist_true, axis=-1)
    bbox_loss = tf.reduce_sum(tf.pow(bbox_true - bbox_pred, 2), axis=None)/b
    return 70*bbox_loss    

def classification(y_true, y_pred):
    b = y_true.shape[0]
    h = y_true.shape[1]
    w = y_true.shape[2]
    c = y_true.shape[3]-5
    
    # sum(sum(-ylog(y'), axis=-1) * exist, axis=None)/ batch_size
    exist_true = tf.cast(y_true[:, :, :, 0], tf.float32)
    class_true = tf.cast(y_true[:, :, :, 5:], tf.float32)
    class_pred = tf.cast(y_pred[:, :, :, 5:] * (1-1e-5) + (1e-5)/2, tf.float32)
    class_loss = tf.reduce_sum(tf.reduce_sum(-class_true*tf.math.log(class_pred), axis=-1) * exist_true, axis=None)/b
    return 10*class_loss


# In[6]:


LR = 1e-4
DROP = 0
BATCH_SIZE = 32
EPOCHS = 15
T_STEPS = int(num_training_data / BATCH_SIZE)

t_gen = train_generator(data_index=data_index, 
                        img_train_path=IMG_TRAIN_PATH, 
                        batch_size=BATCH_SIZE, img_shape=(416, 416, 3), 
                        label_cells=(12, 12))


print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

logs = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

profiler = tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, profile_batch="500,520")


tf.profiler.experimental.start(logs)

model = ImageNet(416, 416, 3, num_of_classes=num_classes)
opt = Adam(learning_rate=LR)
loss_fn = CostFunction()
model(tf.random.normal((1, 416, 416, 3)))
#model.compile(optimizer=opt, loss={"output_1":"binary_crossentropy", "output_2":bbox, "output_3":"categorical_crossentropy"})#, loss=loss_fn, metrics=[existance, bbox, classification])
#model.fit(x=t_gen, epochs=EPOCHS, steps_per_epoch=T_STEPS, verbose=1)
# profiler.set_model(model)
model.fit(t_gen=t_gen, loss_fn=CostFunction(), opt=opt, epochs=EPOCHS, steps_per_epoch=T_STEPS, metrics=[existance, bbox, classification])

tf.profiler.experimental.stop()


# In[8]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir=logs --load_fast=false --port 7777')


# In[7]:


tf.profiler.experimental.stop()


# In[ ]:




