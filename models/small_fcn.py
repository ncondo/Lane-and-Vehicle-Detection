from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, Flatten, MaxPooling2D, Lambda

def small_fcn(input_shape=(64,64,3), train=False):
    model = Sequential()
    # Center and normalize our data
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape, output_shape=input_shape))
    # 1st conv layer with 128 filter, 3x3 each, 50% dropout
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv1',input_shape=input_shape, border_mode="same"))
    model.add(Dropout(0.5))
    # 2nd conv layer with 128 filter, 3x3 each, 50% dropout
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2',border_mode="same"))
    model.add(Dropout(0.5))
    # 3rd conv layer with 128 filter, 3x3 each, 8x8 pooling and dropout
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv3',border_mode="same"))
    model.add(MaxPooling2D(pool_size=(8,8)))
    model.add(Dropout(0.5))
    # This acts like a 128 neuron dense layer
    model.add(Convolution2D(128,8,8,activation="relu",name="dense1"))
    model.add(Dropout(0.5))
    # This is like a 1 neuron dense layer with tanh [-1, 1]
    model.add(Convolution2D(1,1,1,name="dense2", activation="tanh"))
    # Flatten for training
    if train == True:
        model.add(Flatten())

    return model
