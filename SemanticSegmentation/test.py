import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf

img_path = '/home/kaijaz/Python/segmentation/images'
model_file = 'model.h5'

def preprocess(image, img_size):
    # process format that match prediction
    img = cv2.imread(image)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resize = cv2.resize(rgb, (img_size))
    normalized = resize / 255.0
    final = np.expand_dims(normalized, axis = 0)

    return resize, final

def draw_segmentation(ori_image, pred):
    # create mask to draw segmentation
    mask = ori_image.copy()
    gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
    inverse = ~binary.astype(np.uint8)

    # find contour and fill it
    contours, hir = cv2.findContours(binary.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, -1, (255,0,0), cv2.FILLED) # red for object
    contours, hir = cv2.findContours(inverse, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, -1, (0,0,255), cv2.FILLED) # blue for background

    # add up two images
    output = cv2.addWeighted(ori_image, 0.6, mask, 0.4, 0)
    return output

def test(model_file, path):
    # load model
    model = tf.keras.models.load_model(model_file)
    index = np.random.random_integers(0,20)
    filename = os.listdir(path)[index]
    ori, pred = preprocess(os.path.join(img_path, filename), (128,128))
    pred = model.predict(pred)[0]
    output = draw_segmentation(ori, pred)
    plt.imshow(output)
    plt.show()

if __name__ == '__main__':
    test(model_file, img_path)
