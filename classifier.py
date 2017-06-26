import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications


# Dimension of images
img_width, img_height = 64, 64

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 14208
nb_validation_samples = 1776
epochs = 20
batch_size = 16


def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    # Build VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
    bottleneck_features_train = model.predict_generator(
            generator, nb_train_samples // batch_size)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
    bottleneck_features_validation = model.predict_generator(
            generator, nb_validation_samples // batch_size)
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)


def train_top_model():
    train_data = np.load('bottleneck_features_train.npy', encoding='bytes')
    train_labels = np.array([0]*7174 + [1]*7034)

    validation_data = np.load('bottleneck_features_validation.npy', encoding='bytes')
    validation_labels = np.array([0]*897 + [1]*879)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
            metrics=['accuracy'])
    model.fit(train_data, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
