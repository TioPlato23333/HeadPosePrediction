import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from skimage import feature as feat
from skimage import measure
from skimage import filters
from skimage.feature import hog
from skimage.segmentation import active_contour
from skimage.segmentation import chan_vese
from sklearn import metrics
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
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
        contour_image = chan_vese(image, init_level_set='disk')
        binary = np.asarray(contour_image, dtype='float')
        # contours, _= cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # largest_contour = np.reshape(max(contours, key=cv2.contourArea), (-1, 2))
        # largest_contour = np.reshape(contours[0], (-1, 2))
        contour_image = cv2.resize(binary, interpolation=cv2.INTER_AREA, dsize=(int(contour_image.shape[1] / 8), \
            int(contour_image.shape[0] / 8)))
        pos_feat = contour_image.flatten()
        print('[INFO] Position feature dimenstion: ' + str(np.shape(pos_feat)))
        return [], pos_feat

    def createHogFeature(self, image):
        fd = hog(image, feature_vector=True, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        print('[INFO] HOG feature dimenstion: ' + str(np.shape(fd)))
        return fd

    def readFeatureFile(self, file, sample_limit=-1, read_each_row=1):
        print('[INFO] Load feature file...')
        feat = []
        golden = []
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file)
            count = 0
            for row in csv_reader:
                count += 1
                if sample_limit > 0 and count > sample_limit:
                    break
                if count % read_each_row != 0:
                    continue
                feat.append([float(x) for x in row[0].split(',')])
                golden.append([float(x) for x in row[1].split(',')])
        print('[INFO] ' + str(count) + ' rows are read')
        return np.array(feat), np.array(golden)

    def trainAndTest(self, train_file, test_file):
        TRAIN_CLUSTER_SIZE = 144 * 6
        TEST_CLUSTER_SIZE = 8 * 6
        # TRAIN_MAX_LINE = 460800
        # TEST_MAX_LINE = 25600
        CASE_NUM = 520
        print('[INFO] Train and test the feature...')
        with open(train_file) as csv_file_train, open(test_file) as csv_file_test:
            csv_reader_train = csv.reader(csv_file_train)
            csv_reader_test = csv.reader(csv_file_test)
            diff = np.array([])
            for i in range(0, CASE_NUM):
                print('[INFO] Processing case ' + str(i) + '/' + str(CASE_NUM) + '...')
                train_feat = []
                train_golden = []
                count = 0
                for row in csv_reader_train:
                    if count >= TRAIN_CLUSTER_SIZE:
                        break
                    train_feat.append([float(x) for x in row[0].split(',')])
                    train_golden.append([float(x) for x in row[1].split(',')])
                    count += 1
                test_feat = []
                test_golden = []
                count = 0
                for row in csv_reader_test:
                    if count >= TEST_CLUSTER_SIZE:
                        break
                    test_feat.append([float(x) for x in row[0].split(',')])
                    test_golden.append([float(x) for x in row[1].split(',')])
                    count += 1
                # train model
                clf = MultiOutputRegressor(svm.SVR())
                clf.fit(train_feat, train_golden)
                # test model
                result = clf.predict(test_feat)
                product = [np.clip(np.dot(result[i] / np.linalg.norm(result[i]), \
                    test_golden[i] / np.linalg.norm(test_golden[i])), \
                    -1.0, 1.0) for i in range(0, len(result))]
                diff = np.append(diff, np.rad2deg(np.arccos(product)))
            print('[INFO] The prediction error (angle) is ' + str(np.average(diff)) + '(' + str(np.std(diff)) + ')')
            # ALL: [INFO] The prediction error (angle) is 15.225444260273674(10.584194597925089)
            # CONTOUR: [INFO] The prediction error (angle) is 16.32814617323653(11.187172392966717)
            # HOG: [INFO] The prediction error (angle) is 12.397089048993369(8.692957147909784)

    def train(self, feat, golden):
        print('[INFO] Train the feature...')
        n_sample = len(feat)
        if n_sample != len(golden):
            print('[ERROR] Feature size is not consistent with golden (' + str(len(feat)) + ' with ' + \
                str(len(golden)) + ')')
            return
        clf = MultiOutputRegressor(svm.SVR())
        clf.fit(feat, golden)
        with open(self.MODEL_PATH, 'wb') as model_file:
            pickle.dump(clf, model_file)

    def test(self, feat, golden):
        if len(feat) != len(golden):
            print('[ERROR] Feature size is not consistent with golden (' + str(len(feat)) + ' with ' + \
                str(len(golden)) + ')')
            return
        with open(self.MODEL_PATH, 'rb') as model_file:
            clf = pickle.load(model_file)
        result = clf.predict(feat)
        product = [np.clip(np.dot(result[i] / np.linalg.norm(result[i]), golden[i] / np.linalg.norm(golden[i])), \
            -1.0, 1.0) for i in range(0, len(result))]
        diff = np.rad2deg(np.arccos(product))
        sorted_diff = np.sort(diff)
        print('[INFO] The prediction error is ' + str(metrics.mean_squared_error(result, golden)))
        print('[INFO] The prediction error (angle) is ' + str(np.average(diff)) + '(' + str(np.std(diff)) + ')')
        print('[INFO] The best 12800 prediction error (angle) is ' + str(np.average(sorted_diff[0: 12800])) + \
            '(' + str(np.std(sorted_diff[0: 12800])) + ')')
        print('[INFO] The worst 2000 prediction error (angle) is ' + str(np.average(sorted_diff[-2000:])) + \
            '(' + str(np.std(sorted_diff[-2000:])) + ')')
        # plt.hist(diff, bins=20)
        # plt.show()
        return np.argsort(diff), result

    def extractFeatureToFile(self, file_path, max_sample_num=-1):
        # generate feature csv
        with open(FEATURE_PATH, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            list_len = len(database.image_list)
            if max_sample_num > 0:
                list_len = max_sample_num
            for index in range(0, list_len):
                print('[INFO] Progress: ' + str(index) + '/' + str(list_len))
                image, golden_feat = database.loadImage(index)
                # downsample image
                contour, pos_feat = database.createPositionFeature(image)
                hog_feat = database.createHogFeature(image)
                current_feat = np.append(pos_feat, hog_feat)
                writer.writerow([','.join([str(x) for x in current_feat]), ','.join([str(x) for x in golden_feat[0: 3]])])

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
    MODEL_PATH = 'model.pkl'
    # private variables
    image_list = []

if __name__ == '__main__':
    DATABASE_PATH = '../s00-09'
    FEATURE_PATH = 'feature.csv'
    LOAD_FEATURE_PATH = '../s00-09/test_feature5.csv'
    LOAD_FEATURE_PATH2 = '../s00-09/synth_feature5.csv'
    # PREDICTION_PATH = 'prediction.csv'
    database = DataBase(DATABASE_PATH)
    database.extractFeatureToFile(FEATURE_PATH)
    # database.trainAndTest(LOAD_FEATURE_PATH2, LOAD_FEATURE_PATH)
    # train/test
    # feat_train, golden1 = database.readFeatureFile(LOAD_FEATURE_PATH2, 2000)
    # feat_test, golden2 = database.readFeatureFile(LOAD_FEATURE_PATH)
    # database.train(feat_train, golden1)
    # index, result = database.test(feat_test, golden2)
    '''
    # test codes
    SAMPLE_SHOW = 20000
    temp_index = index[SAMPLE_SHOW]
    temp_result = result[temp_index]
    image, golden_feat = database.loadImage(temp_index)
    contour, _ = database.createPositionFeature(image)
    database.showImageIn3dPlot(image, golden_feat, contour, temp_result)
    '''
    sys.exit()
