import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataPreprocess:
    def __init__(self, img_size, nb_channels, batch_size, validation_split):
        self.img_size = img_size
        self.nb_channels = nb_channels
        self.batch_size = batch_size
        self.validation_split = validation_split

        # thresholds for HSV region of intrest detection
        self.lower = np.array([0, 100, 100], dtype="uint8")
        self.upper = np.array([35, 255, 255], dtype="uint8")

        # data augmentation methods
        self.train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=180,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            fill_mode="nearest",
            validation_split=self.validation_split
        )
        # image data generator for test set doesn't require augmentations
        self.test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    def display(self, image, img_name):
        plt.figure(figsize=(15, 10))
        plt.imshow(image)
        plt.title(img_name)
        plt.show()

    # function to remove incorrectly labelled images and outliers
    def remove_noise(self, image_path, flag=False):
        img_bgr = cv2.imread(image_path)
        img_blur = cv2.GaussianBlur(img_bgr.copy(), (1, 1), 10)
        img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

        # Step to increase brightness for fire detection
        if flag:
            img_hsv[:, :, 2] += 50

        mask = cv2.inRange(img_hsv, self.lower, self.upper)

        if np.any(mask):
            return 1
        return 0

    def refine_train_data(self, train_dir, train_refined_dir):
        train_fire = os.listdir(train_dir + "Fire/")
        train_no_fire = os.listdir(train_dir + "No_Fire/")
        file_path = []
        types = []

        for i in train_fire:
            inp_path = train_dir + "Fire/" + i
            if self.remove_noise(inp_path, flag=True):
                file_path.append(inp_path)
                types.append('Fire')

        for i in train_no_fire:
            inp_path = train_dir + "No_Fire/" + i
            if ~self.remove_noise(inp_path):
                file_path.append(inp_path)
                types.append('No_Fire')

        df = pd.DataFrame(data={'file_path': file_path, 'type': types})

        # Remove and create folder structure for refined train set
        if os.path.exists(train_refined_dir):
            shutil.rmtree(train_refined_dir, ignore_errors=False)
        os.mkdir(train_refined_dir)
        os.mkdir(os.path.join(train_refined_dir, "Fire"))
        os.mkdir(os.path.join(train_refined_dir, "No_Fire"))

        for i in df['file_path'].values:
            if 'No_Fire' in i:
                shutil.copy(i, os.path.join(train_refined_dir, "No_Fire"))
            else:
                shutil.copy(i, os.path.join(train_refined_dir, "Fire"))

        return train_refined_dir

    def image_augmentation(self, train_refined_dir, test_dir):
        train_generator = self.train_datagen.flow_from_directory(train_refined_dir,
                                                                 batch_size=self.batch_size,
                                                                 target_size=(self.img_size, self.img_size),
                                                                 color_mode="rgb",
                                                                 shuffle=True,
                                                                 class_mode='binary',
                                                                 subset='training'
                                                                 )
        val_generator = self.train_datagen.flow_from_directory(train_refined_dir,
                                                               batch_size=self.batch_size,
                                                               target_size=(self.img_size, self.img_size),
                                                               color_mode="rgb",
                                                               shuffle=True,
                                                               class_mode='binary',
                                                               subset='validation'
                                                               )
        test_generator = self.test_datagen.flow_from_directory(test_dir,
                                                               batch_size=self.batch_size,
                                                               target_size=(self.img_size, self.img_size),
                                                               color_mode="rgb",
                                                               shuffle=False,
                                                               class_mode='binary'
                                                               )
        return train_generator, val_generator, test_generator

    def view_imagegen_samples(self, images_arr):
        fix, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
