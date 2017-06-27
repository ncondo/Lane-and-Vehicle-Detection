import glob
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import io
from skimage.transform import resize
from moviepy.editor import VideoFileClip
from models import small_fcn


if __name__=="__main__":

    cars = glob.glob("./data/train/vehicles/*.png")
    non_cars = glob.glob("./data/train/non-vehicles/*.png")

    # Read X Vector
    X = []
    for file in cars:
        X.append(io.imread(file))
    for file in non_cars:
        X.append(io.imread(file))
    X = np.array(X)

    # Generate Y Vector
    Y = np.concatenate([np.ones(len(cars)), np.zeros(len(non_cars))])

    # Split train and validation dataset with 10%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # Show messages
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    model = small_fcn.small_fcn(train=True)
    model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=64, nb_epoch=20, verbose=2, validation_data=(X_test, Y_test))

    model.save_weights('./models/model.h5')
