##########
# Examples
##########
# from keras.layers import Conv2D

# 200 by 200 grayscale image, 16 filters with width and height 2, stride of 2 and no padding, Relu activation function
# Conv2D(filters=16, kernel_size=2, strides=2, activation='relu', input_shape=(200, 200, 1))

# 32 filters with width and height 3, with padding, Relu activation function
# Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')

# Just another format for Convolutional layers in Keras
# 64 filters of size 2x2 all other values are default
# Conv2D(64, (2,2), activation='relu')

# Number of parameters in convolutional layer
# K - the number of filters in the convolutional layer
# F - the height and width of the convolutional filters
# D_in - the depth of the previous layer
# numParams =  K*F*F*D_in + K

# Shape of a Convolutional Layer
# K - the number of filters in the convolutional layer
# F - the height and width of the convolutional filters
# S - the stride of the convolution
# H_in - the height of the previous layer
# W_in - the width of the previous layer
# If padding = 'same'
# height = ceil(float(H_in) / float(S))
# width = ceil(float(W_in) / float(S))
# Else if padding = 'valid'
# height = ceil(float(H_in - F + 1) / float(S))
# width = ceil(float(W_in - F + 1) / float(S))

####################################################################################################

# from keras.models import Sequential
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D

# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', 
#     activation='relu', input_shape=(128, 128, 3)))
# model.summary()

####################################
# For Pooling layer
# You must include the following argument:

# pool_size - Number specifying the height and width of the pooling window.
# There are some additional, optional arguments that you might like to tune:

# strides - The vertical and horizontal stride. If you don't specify anything, strides will default to pool_size.
# padding - One of 'valid' or 'same'. If you don't specify anything, padding is set to 'valid'.
# NOTE: It is possible to represent both pool_size and strides as either a number or a tuple.

# MaxPooling2D(pool_size, strides, padding)

from keras.models import Sequential
from keras.layers import MaxPooling2D

model = Sequential()
model.add(MaxPooling2D(pool_size=2, strides=2, input_shape=(100, 100, 15)))
model.summary()

####################################