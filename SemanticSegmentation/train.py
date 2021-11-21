import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_loader import DataGenerator
from Unet import build_unet
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

img_path = 'images'
mask_path = 'annotations/trimaps'

def read_data(path):
    filename = [file.split('.')[0] for file in os.listdir(path)]
    x = np.array([name + '.jpg' for name in filename])
    y = np.array([name + '.png' for name in filename])
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.1, random_state = 15)
    return x_train, x_valid, y_train, y_valid

def build_model(img_size, num_classes, lr):
    # build model
    model = build_unet(img_size, num_classes)
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)
    return model

def train(batch_size, img_size, shuffle, lr, epochs, num_classes):
    x_train, x_valid, y_train, y_valid = read_data(img_path)
    model = build_model(img_size, num_classes, lr)

    # define generator
    train_generator = DataGenerator(img_path, mask_path, x_train, y_train, batch_size, shuffle, img_size[:2])
    val_generator = DataGenerator(img_path, mask_path, x_valid, y_valid, batch_size, shuffle, img_size[:2])
    total_train = len(x_train)

    # callbacks
    callbacks = [
        ModelCheckpoint("model.h5", verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=5, verbose=1)
    ]

    # train
    model.fit_generator(
        generator = train_generator,
        validation_data = val_generator,
        steps_per_epoch = total_train//batch_size,
        epochs = epochs,
        callbacks = callbacks
    )

if __name__ == '__main__':
    batch_size = 16
    img_size = (128,128,3)
    lr = 1e-3
    shuffle = True
    num_classes = 3
    epochs = 10
    train(batch_size, img_size, shuffle, lr, epochs, num_classes)
