# parts of the code were adapted from https://github.com/jemiar/surgery-gesture-recog (for surgical gesture recognition experiment consult this link)
# parts of the code were adapted from https://github.com/yashkant/padam-tensorflow(for image classification experiment consult this link)

#https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/training/adam.py"""

#Our proposed_ASGD_adam_version and proposed_ASGD_amsgrad_version located in the file above can easily be included in both links

#the following code should be added to make the source code compatible with tensorflow version 1

#import tensorflow.compat.v1 as tf

#tf.disable_v2_behavior()

#tf.compat.v1.disable_eager_execution()

#from keras.utils import to_categorical

# tf.train.get_slope() should be excluded

#import keras
