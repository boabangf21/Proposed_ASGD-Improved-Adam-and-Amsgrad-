# parts of the code were adapted from https://github.com/jemiar/surgery-gesture-recog (for surgical gesture recognition experiment consult this link)
# parts of the code were adapted from https://github.com/yashkant/padam-tensorflow(for image classification experiment consult this link)
the following code should be added to make the source code compatible with tensorflow version 1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
from keras.utils import to_categorical
import keras
