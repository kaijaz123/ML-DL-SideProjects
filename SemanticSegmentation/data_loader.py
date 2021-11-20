import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

class DataGenerator(keras.utils.Sequence):
    def __init__(self, img_path, mask_path, images, masks, batch_size = 32, shuffle = True, img_size = (128,128)):
        self.img_path = img_path
        self.mask_path = mask_path
        self.images = images
        self.masks = masks
        self.shuffle = shuffle
        self.img_size = img_size
        self.batch_size = batch_size
        self.width, self.height = self.img_size

    def read_img(self, img_file, img_path):
        img = cv2.imread(os.path.join(img_path,img_file))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resize = cv2.resize(rgb, (self.width, self.height))
        resize = resize / 255.0
        return resize

    def read_mask(self, mask_file, mask_path):
        mask = cv2.imread(os.path.join(mask_path,mask_file), cv2.IMREAD_GRAYSCALE) # gray scale for tf one hot encode purpose
        mask = mask - 1
        resize = cv2.resize(mask, (self.width, self.height))
        return resize

    def __len__(self):
        return len(self.images)//self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            order = np.random.permutation(len(self.images))
            self.images = self.images[order]
            self.masks = self.masks[order]

    def __getitem__(self, index):
        # generate one batch of data
        img_list = self.images[index:index+self.batch_size]
        mask_list = self.masks[index:index+self.batch_size]
        x, y = self.__data_generation(img_list, mask_list, self.img_path, self.mask_path)

        return x, y

    def __data_generation(self, img_list, mask_list, img_path, mask_path):
        x = []
        y = []
        for image,mask in zip(img_list,mask_list):
            img = self.read_img(image, img_path)
            mask = self.read_mask(mask, mask_path)
            x.append(img)
            y.append(mask)

        # transform into array
        x = np.array(x)
        y = tf.one_hot(np.array(y), 3) # perform one hot encoding on whole array - 3 pixel values 0, 1, and 2 so 3 classes
        y.set_shape([self.batch_size,self.width, self.height, 3]) # reshape because tf one hot will generate one more 3 channels for image
        return x, y
