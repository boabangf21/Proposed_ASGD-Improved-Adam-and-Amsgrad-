from __future__ import absolute_import, division, print_function
import os
import sys
print(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]
#from tensorflow.contrib.opt import AdamWOptimizer
import tensorflow_addons as tfa
import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
import numpy as np
from keras.datasets import cifar10
import keras.callbacks as callbacks
#import keras.utils.np_utils as kutils
from tensorflow.keras import utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from wide_resnet import WRNModel
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import h5py
#from keras.utils import plot_model
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()




sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

print(sys.path)
from padam import Padam
from amsgrad import AMSGrad
from proposed_ASGD_amsgrad_version import Proposed_ASGD_amsgrad_version
from proposed_ASGD_adam_version import Proposed_ASGD_adam_version

dataset = 'cifar10'
# Model is saved is 'model_{optim}_{dataset}_epochs{X}.h5' where X = continue_epoch
# Csv file is saved as 'log_{optim}_{dataset}.h5'


if dataset == 'cifar10':
    MEAN = [0.4914, 0.4822, 0.4465]
    STD_DEV = [0.2023, 0.1994, 0.2010]
    from keras.datasets import cifar10
    (trainX, trainY), (testX, testY) = cifar10.load_data()
elif dataset == 'cifar100':
    MEAN = [0.507, 0.487, 0.441]
    STD_DEV = [0.267, 0.256, 0.276]
    from keras.datasets import cifar100
    (trainX, trainY), (testX, testY) = cifar100.load_data()    



def preprocess(t):
    paddings = tf.constant([[2, 2,], [2, 2],[0,0]])
    t = tf.pad(t, paddings, 'CONSTANT')
    t = tf.image.random_crop(t, [32, 32, 3])
    t = normalize(t) 
    return t


def normalize(t):
    t = tf.div(tf.subtract(t, MEAN), STD_DEV) 
    return t

def save_model(filepath, model):
    file = h5py.File(filepath,'w')
    weight = model.get_weights()
    for i in range(len(weight)):
        file.create_dataset('weight'+str(i),data=weight[i])
    file.close()

def load_model(filepath, model):
    file=h5py.File(filepath,'r')
    weight = []
    for i in range(len(file.keys())):
        weight.append(file['weight'+str(i)][:])
    model.set_weights(weight)
    return model 

hyperparameters = {
    'cifar10': {
        'epoch':200,
        'batch_size': 256,
        'decay_after': 50,
        'classes': 10
    },
    'cifar100': {
        'epoch': 200,
        'batch_size': 256,
        'decay_after': 50,
        'classes': 100 
    },
    'imagenet': {
        'epoch': 100,
        'batch_size': 64,
        'decay_after': 30
    }
}

optim_params = {
    
    
    'proposed_ASGD_amsgrad_version': {
        'weight_decay': 0.0001,
        'lr_min': 0.001,
        'lr_max':0.01,
        #'lr': 0, # dummy value
        'b1': 0.9,
        'b2': 0.999,
        'color': 'red',
        'linestyle':'-'
    },
    'proposed_ASGD_adam_version': {
        'weight_decay': 0.0001,
        'lr_min': 0.001,
        'lr_max':0.01,
        #'lr': 0, # dummy value
        'b1': 0.9,
        'b2': 0.999,
        'color': 'red',
        'linestyle':'-'
    },  
    
    
    'padam': {
        'weight_decay': 0.0005,
        'lr': 0.1,
        'p': 0.125,
        'b1': 0.9,
        'b2': 0.999,
        'color': 'darkred',
        'linestyle':'-'
    },
    'adam': {
        'weight_decay': 0.0001,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'color': 'orange',
        'linestyle':'--'
    },
    'adamw': {
        'weight_decay': 0.025,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'color': 'magenta',
        'linestyle':'--'
    },
    'amsgrad': {
        'weight_decay': 0.0001,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.99,
        'color' : 'darkgreen',
        'linestyle':'-.'
    },
    'sgd': {
        'weight_decay': 0.0005,
        'lr': 0.1,
        'm': 0.9,
        'color': 'blue',
        'linestyle':'-'
    }
}

hp = hyperparameters[dataset]
epochs = hp['epoch']
batch_size = hp['batch_size']
#classes = hp['classes']

img_rows, img_cols = 32, 32
train_size = trainX.shape[0]

trainX = trainX.astype('float32')
trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
testX = testX.astype('float32')
testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))
trainY = utils.to_categorical(trainY)
testY = utils.to_categorical(testY)
#tf.train.create_global_step()



# +

def random_crop(img, random_crop_size):
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]
     

def crop_generator(batches, crop_length, num_channel = 3):
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, num_channel))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)

        
        
datagen_train = ImageDataGenerator(samplewise_center=True,
                             zca_whitening=True,
                             horizontal_flip=True
                            )
     
CROP_SIZE=32
datagen_train.fit(trainX, augment=True)
     

train_flow = datagen_train.flow(trainX, trainY, batch_size=batch_size)
train_flow_w_crops = crop_generator(train_flow, CROP_SIZE)
valid_flow = datagen_train.flow(trainX, trainY, batch_size=batch_size)       




#CROP_SIZE=32
#datagen_train = ImageDataGenerator(zca_whitening=True,
  #                           horizontal_flip=True, validation_split=0.2)
#datagen_train.fit(trainX, seed=0, augment=True)
#train_flow = datagen_train.flow(trainX, trainY, batch_size=batch_size, subset="training")
#train_ds, val_ds, test_ds = get_dataset_partitions_tf(train_ds_1)

#
#train_flow_w_crops = crop_generator(train_flow, CROP_SIZE)
#val_flow= datagen_train.flow(trainX, trainY, batch_size=batch_size, subset="validation")


#from sklearn.model_selection import train_test_split
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration for creating new images
#datagen_train = ImageDataGenerator(
 #   rotation_range=20,
   # width_shift_range=5./30,
  #  height_shift_range=5./30,
  #  brightness_range=None,
  #  shear_range=5./30,
   # zoom_range=5./30,
    #channel_shift_range=0.0,
    #fill_mode=&#x27;nearest',
   # cval=0.0,
   # horizontal_flip=True,
   # vertical_flip=True,
  # )
#datagen_train.fit(trainX, seed=0, augment=True)


#trainX, validationX, trainY, validationY = train_test_split(trainX, trainY, test_size=0.3, random_state=0)
#datagen_train.fit(trainX, seed=0, augment=True)
# -

epochs=200

# +
optim_array = [ 'proposed_ASGD_amsgrad_version']
#optim_array=['proposed_ASGD']

history = {}
# -


import tensorflow as tf
from tensorflow.keras.layers import RandomFlip, RandomRotation
from tensorflow.python import keras
for i in range(4):
    
    if(i != 0):
        continue_training = True # Flag to continue training   
        continue_epoch = (i)*10
    else:
        continue_training = False
        #op['lr_min']=0.001
        #op['lr_max']=0.01

    for optimizer in optim_array:
        #'lr':0.001,
        print('-'*40, optimizer, '-'*40)
        op = optim_params[optimizer]
        #op['lr'] = op['lr']/(10**i) 
        op['lr_min'] = op['lr_min']/(10**i) 
        op['lr_max'] = (op['lr_max'])/(10**i)
        #model._set_inputs(tf.zeros((batch_size, 32, 32, 3)))
       # model= tf.keras.layers.RandomFlip("horizontal")(model)
       # model= tf.keras.layers.GaussianNoise(5.0)(model)# try 10
       # model = tf.keras.layers.RandomZoom(0.15)(model)
       # model = tf.keras.layers.RandomTranslation(0.1, 0.1)(model)
       # model = tf.keras.layers.RandomRotation(0.15)(model)
        
        if optimizer == 'adamw' and dataset=='imagenet':
            op['weight_decay'] = 0.05 

        if optimizer is not 'adamw':
            model = WRNModel(depth=16, classes=10, multiplier=4, wd = op['weight_decay'], dropout_rate=0.3)
        else:
            model = WRNModel( depth=16, classes=10,  multiplier=4, wd = 0)


      #  model._set_inputs(tensorflow.python.keras.layers.RandomFlip("horizontal"))
       # model._set_inputs(tf.keras.layers.GaussianNoise(5.0))# try 10
       # model._set_inputs(tf.keras.layers.RandomZoom(0.15))
       # model._set_inputs( tf.keras.layers.RandomTranslation(0.1, 0.1))
       # model._set_inputs(tf.keras.layers.RandomRotation(0.15))
        logfile = 'log_'+optimizer+ '_256_augment' + dataset +'.csv'

        if(continue_training):
            load_model_filepath = 'model_'+optimizer+'_'  + dataset + '_epochs'+ str(continue_epoch)+'.h5'
            save_model_filepath = 'model_'+optimizer+'_'  + dataset + '_epochs'+ str(continue_epoch+epochs)+'.h5'
            model = load_model(load_model_filepath, model)
        else:
            save_model_filepath = 'model_'+optimizer+'_256_augment'  + dataset + '_epochs'+ str(epochs)+'.h5'

       # learning_rate = tf.compat.v1.train.exponential_decay(op['lr'],  batch_size,
                                       #    hp['decay_after']*train_size, 0.1, staircase=True)
        
    
        learning_rate_min = tf.compat.v1.train.exponential_decay(op['lr_min'],  batch_size,
                                           hp['decay_after']*train_size, 0.1, staircase=True)
        
        learning_rate_max=tf.compat.v1.train.exponential_decay(op['lr_max'], batch_size,
                                           hp['decay_after']*train_size, 0.1, staircase=True)
        if optimizer == 'proposed_ASGD_amsgrad_version':
            optim = Proposed_ASGD_amsgrad_version(learning_rate_min=learning_rate_min,learning_rate_max=learning_rate_max, beta1=op['b1'], beta2=op['b2'])
        elif optimizer == 'proposed_ASGD_adam_version':
            optim = Proposed_ASGD_adam_version(learning_rate_min=learning_rate_min,learning_rate_max=learning_rate_max, beta1=op['b1'], beta2=op['b2'])
        elif optimizer == 'padam':
            optim = Padam(learning_rate=learning_rate, p=op['p'], beta1=op['b1'], beta2=op['b2'])
        elif optimizer == 'adam':
            optim = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=op['b1'], beta2=op['b2'])
        elif optimizer == 'super':
            # adamw = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
            #optim = tf.contrib.opt.AdamWOptimizer(weight_decay=op['weight_decay'], learning_rate=learning_rate,  beta1=op['b1'], beta2=op['b2'])
            optim = Super(learning_rate=learning_rate,  beta1=op['b1'], beta2=op['b2'])
        elif optimizer == 'wada':
            # adamw = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
            #optim = tf.contrib.opt.AdamWOptimizer(weight_decay=op['weight_decay'], learning_rate=learning_rate,  beta1=op['b1'], beta2=op['b2'])
            optim = Wada(learning_rate=learning_rate,  beta1=op['b1'], beta2=op['b2'])
        elif optimizer == 'amsgrad':
            optim = AMSGrad(learning_rate=learning_rate, beta1=op['b1'], beta2=op['b2'])
        elif optimizer == 'sgd':
            optim = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=op['m'])

       # model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'], global_step=tf.train.get_global_step())
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
        csv_logger = CSVLogger(logfile, append=True, separator=';')


        history[optimizer] = model.fit_generator(train_flow_w_crops, epochs = epochs,  steps_per_epoch=len(trainX) / batch_size,
                                     validation_data=(valid_flow), validation_steps=(len(trainX) / batch_size), verbose=1, callbacks = [csv_logger])

        scores = model.evaluate_generator((valid_flow ) , verbose=1)

        #print("Final test loss and accuracy:", scores)
        #save_model(save_model_filepath, model)




plt.figure(1)
        for optimizer in optim_array:
            op = optim_params[optimizer]
            train_loss = history[optimizer].history['loss']
            epoch_count = range(1, len(train_loss) + 1)
            plt.plot(epoch_count, train_loss, color=op['color'], linestyle=op['linestyle'])
        plt.legend(optim_array)
        plt.xlabel('Epochs')
        plt.ylabel('Train Loss')
        plt.savefig('figure_'+dataset+'_train_loss.png')
        #test plot
        plt.figure(2)
        for optimizer in optim_array:
            op = optim_params[optimizer]
            test_error = []
            for i in history[optimizer].history['val_acc']:
                test_error.append(1-i)
                epoch_count = range(1, len(test_error) + 1)
            plt.plot(epoch_count, test_error, color=op['color'], linestyle=op['linestyle'])
        plt.legend(optim_array)
        plt.xlabel('Epochs')
        plt.ylabel('Test Error')
        plt.figure(3)
        for optimizer in optim_array:
            op = optim_params[optimizer]
            test_error = []
            for i in history[optimizer].history['val_top_k_categorical_accuracy']:
                test_error.append(1-i)
                epoch_count = range(1, len(test_error) + 1)
            plt.plot(epoch_count, test_error, color=op['color'], linestyle=op['linestyle'])
        plt.legend(optim_array)
        plt.xlabel('Epochs')
        plt.ylabel('Test Error')   
        plt.savefig('figure_'+dataset+'_test_error_top_5.png')

optim_array = ['proposed_ASGD','amsgrad', 'sgd', 'padam', 'adam']

# +
import matplotlib.pyplot as plt 
import csv 
import pandas as pd
  

df=pd.read_csv('/users/boabangfrancis/padam-tensorflow-master/padam-tensorflow-master/wide-resnet/log_adam_cifar100.csv','r') 
df
# -

# pwd


#test plot
plt.figure(2)
for optimizer in optim_array:
    op = optim_params[optimizer]
    test_error = []
    for i in history[optimizer].history['val_acc']:
        test_error.append(1-i)
    epoch_count = range(1, len(test_error) + 1)
    plt.plot(epoch_count, test_error, color=op['color'], linestyle=op['linestyle'])
plt.legend(optim_array)
plt.xlabel('Epochs')
plt.ylabel('Test Error')

# plt.show()
plt.savefig('figure_'+dataset+'_test_error_top_1.png')

#test plot
plt.figure(3)
for optimizer in optim_array:
    op = optim_params[optimizer]
    test_error = []
    for i in history[optimizer].history['val_top_k_categorical_accuracy']:
        test_error.append(1-i)
    epoch_count = range(1, len(test_error) + 1)
    plt.plot(epoch_count, test_error, color=op['color'], linestyle=op['linestyle'])
plt.legend(optim_array)
plt.xlabel('Epochs')
plt.ylabel('Test Error')

plt.savefig('figure_'+dataset+'_test_error_top_5.png')
