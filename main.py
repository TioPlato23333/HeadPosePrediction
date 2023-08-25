import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import feature as feat
from skimage import measure
from skimage import filters
from skimage.feature import hog
from skimage.segmentation import active_contour
from skimage.segmentation import chan_vese
import sys
import zipfile

class DataBase:
    def __init__(self, database_path):
        print('[INFO] Loading database image list...')
        for subject_name in os.listdir(database_path):
            subject_path = os.path.join(database_path, subject_name)
            if os.path.isdir(subject_path):
                test_path = os.path.join(subject_path, self.TEST_PATH)
                for file_name in os.listdir(test_path):
                    file_path = os.path.join(test_path, file_name)
                    if os.path.isfile(file_path) and file_path.endswith('.csv'):
                        # read zip file
                        file_stem = os.path.splitext(file_name)[0]
                        zip_file_path = os.path.join(test_path, file_stem + self.ZIP_EXTENSION)
                        with open(file_path) as csv_file:
                            # read csv file
                            csv_reader = csv.reader(csv_file)
                            row_count = 0
                            for row in csv_reader:
                                image_file_name = self.IMAGE_FILE_FORMAT.format(row_count) + self.IMAGE_EXTENSION
                                feature = [float(x) for x in row[0: -1]]
                                self.image_list.append({'zip': zip_file_path, 'name': image_file_name, \
                                    'feature': feature})
                                row_count += 1

    def showImageIn3dPlot(self, image, feature, contour=[]):
        fig = plt.figure()
        width = image.shape[1]
        height = image.shape[0]
        # show 2D plot
        ax1 = fig.add_subplot(221)
        ax1.imshow(image, cmap='gray')
        plt.arrow(width / 2.0, height / 2.0, feature[0] * self.DISPLAY_COEFFICIENT, feature[1] * self.DISPLAY_COEFFICIENT, \
            width=self.DISPLAY_ARROW_SIZE)
        # show 3D plot
        ax2 = fig.add_subplot(222, projection='3d')
        xx, yy = np.meshgrid(np.linspace(0, width, width), np.linspace(0, height, height))
        ax2.quiver(width / 2.0, height / 2.0, self.DISPLAY_OFFSET, feature[0] * self.DISPLAY_COEFFICIENT, \
            feature[1] * self.DISPLAY_COEFFICIENT, feature[2] * self.DISPLAY_COEFFICIENT)
        ax2.contourf(xx, yy, image, zdir='z', offset=self.DISPLAY_OFFSET, cmap='gray')
        # show chan vese result
        ax3 = fig.add_subplot(223)
        ax3.imshow(chan_vese(image), cmap='gray')
        # show key points
        if len(contour) > 0:
            print('[INFO] Contour points number: ' + str(len(contour)))
            ax3.plot(contour[:, 0], contour[:, 1], '-r', lw=3)
        plt.show()

    def loadImage(self, index=0):
        if index > len(self.image_list):
            print('[ERROR] Index is out of the boundary of the image list.')
            sys.exit(-1)
        zip_file_path = self.image_list[index]['zip']
        image_file_name = self.image_list[index]['name']
        feature = self.image_list[index]['feature']
        with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            print('[INFO] Open image file: ' + image_file_name)
            zip_image = zip_file.read(image_file_name)
            image = cv2.imdecode(np.frombuffer(zip_image, np.uint8), cv2.IMREAD_GRAYSCALE)
            print('[INFO] Image feature: ' + ', '.join([str(x) for x in feature]))
            return image, feature

    def extractKeyPoints(self, image):
        # find contour
        contour_image = chan_vese(image)
        binary = np.asarray(contour_image, dtype='uint8')
        contours, _= cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = np.reshape(max(contours, key=cv2.contourArea), (-1, 2))
        # largest_contour = np.reshape(contours[0], (-1, 2))
        contour_image = cv2.resize(binary, dsize=(int(contour_image.shape[1] / 2), int(contour_image.shape[0] / 2)))
        pos_feat = contour_image.flatten()
        print('[INFO] Position feature dimenstion: ' + str(np.shape(pos_feat)))
        return largest_contour, pos_feat

    def createHogFeature(self, image):
        fd = hog(image, feature_vector=True)
        print('[INFO] HOG feature dimenstion: ' + str(np.shape(fd)))

    # constant variables
    # database setting
    TEST_PATH = 'test'
    IMAGE_FILE_FORMAT = '{:08d}'
    IMAGE_EXTENSION = '.bmp'
    ZIP_EXTENSION = '.zip'
    # display setting
    DISPLAY_OFFSET = 100
    DISPLAY_COEFFICIENT = 100
    DISPLAY_ARROW_SIZE = 3
    # private variables
    image_list = []

if __name__ == '__main__':
    DATABASE_PATH = '../s00-09'
    database = DataBase(DATABASE_PATH)
    image, feature = database.loadImage(99)
    contour, pos_feat = database.extractKeyPoints(image)
    database.createHogFeature(image)
    database.showImageIn3dPlot(image, feature, contour)
    # test codes
    sys.exit()
