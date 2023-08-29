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
from sklearn import metrics
from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor
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

    def showImageIn3dPlot(self, image, feature, contour=[], predict_feat=[]):
        fig = plt.figure()
        width = image.shape[1]
        height = image.shape[0]
        # show 2D plot
        ax1 = fig.add_subplot(221)
        ax1.imshow(image, cmap='gray')
        plt.arrow(width / 2.0, height / 2.0, feature[0] * self.DISPLAY_COEFFICIENT, feature[1] * self.DISPLAY_COEFFICIENT, \
            width=self.DISPLAY_ARROW_SIZE)
        if len(predict_feat) > 0:
            plt.arrow(width / 2.0, height / 2.0, predict_feat[0] * self.DISPLAY_COEFFICIENT, \
                predict_feat[1] * self.DISPLAY_COEFFICIENT, \
                width=self.DISPLAY_ARROW_SIZE, color='red')
        # show 3D plot
        ax2 = fig.add_subplot(222, projection='3d')
        xx, yy = np.meshgrid(np.linspace(0, width, width), np.linspace(0, height, height))
        ax2.quiver(width / 2.0, height / 2.0, self.DISPLAY_OFFSET, feature[0] * self.DISPLAY_COEFFICIENT, \
            feature[1] * self.DISPLAY_COEFFICIENT, feature[2] * self.DISPLAY_COEFFICIENT)
        if len(predict_feat) > 0:
            ax2.quiver(width / 2.0, height / 2.0, self.DISPLAY_OFFSET, predict_feat[0] * self.DISPLAY_COEFFICIENT, \
                predict_feat[1] * self.DISPLAY_COEFFICIENT, predict_feat[2] * self.DISPLAY_COEFFICIENT, color='red')
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

    def createPositionFeature(self, image):
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
        return fd

    def readFeatureFile(self, file):
        print('[INFO] Load feature file...')
        feat = []
        golden = []
        SAMPLE_LIMIT = 1000
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file)
            count = 0
            for row in csv_reader:
                if count >= SAMPLE_LIMIT:
                    break
                feat.append([float(x) for x in row[0].split(',')])
                golden.append([float(x) for x in row[1].split(',')])
                count += 1
        return np.array(feat), np.array(golden)

    def trainAndTest(self, feat, golden):
        print('[INFO] Train and test the feature...')
        n_sample = len(feat)
        if n_sample != len(golden):
            print('[ERROR] Feature size is not consistent with golden (' + str(len(feat)) + ' with ' + \
                str(len(golden)) + ')')
            return
        n_train = int(n_sample * self.TRAIN_PERCENTAGE)
        clf = MultiOutputRegressor(svm.SVR())
        clf.fit(feat[0: n_train], golden[0: n_train])
        result = clf.predict(feat[n_train: n_sample])
        error = metrics.mean_squared_error(result, golden[n_train: n_sample])
        print('[INFO] The prediction error is ' + str(error))
        return range(n_train, n_sample), result

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
    # SVM parameter
    TRAIN_PERCENTAGE = 0.3
    # private variables
    image_list = []

if __name__ == '__main__':
    DATABASE_PATH = '../s00-09'
    FEATURE_PATH = 'feature.csv'
    LOAD_FEATURE_PATH = 'important_feature.csv'
    PREDICTION_PATH = 'prediction.csv'
    database = DataBase(DATABASE_PATH)
    '''
    # generate feature csv
    with open(FEATURE_PATH, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file);
        for index in range(0, len(database.image_list)):
        # for index in range(0, len(database.image_list)):
            image, golden_feat = database.loadImage(index)
            contour, pos_feat = database.createPositionFeature(image)
            hog_feat = database.createHogFeature(image)
            current_feat = np.append(pos_feat, hog_feat)
            writer.writerow([','.join([str(x) for x in current_feat]), ','.join([str(x) for x in golden_feat[0: 3]])])
    '''
    feat, golden = database.readFeatureFile(LOAD_FEATURE_PATH)
    index, result = database.trainAndTest(feat, golden)
    '''
    # generate prediction csv
    with open(PREDICTION_PATH, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file);
        for i in range(0, len(index)):
            writer.writerow([index[i], ','.join([str(x) for x in result[i]])])
    '''
    SAMPLE_SHOW = 101
    temp_index = index[SAMPLE_SHOW]
    temp_result = result[SAMPLE_SHOW]
    image, golden_feat = database.loadImage(temp_index)
    contour, _ = database.createPositionFeature(image)
    # test codes
    database.showImageIn3dPlot(image, golden_feat, contour, temp_result)
    sys.exit()
