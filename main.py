import csv
import cv2
from enum import Enum
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
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import sys
import zipfile

class DataBaseType:
    TEST_TYPE = 'test'
    SYNTH_TYPE = 'synth'
    EXTRACT_FEAT_MODE = 1
    TRAIN_TEST_MODE = 2
    TRAIN_ONLY_MODE = 3
    TEST_ONLY_MODE = 4
    VISUALIZATION_MODE = 5

class DataBase:
    def __init__(self, database_path, database_type=DataBaseType.TEST_TYPE):
        print('[INFO] Loading database image list...')
        for subject_name in os.listdir(database_path):
            subject_path = os.path.join(database_path, subject_name)
            if os.path.isdir(subject_path):
                case_path = os.path.join(subject_path, database_type)
                for file_name in os.listdir(case_path):
                    file_path = os.path.join(case_path, file_name)
                    if os.path.isfile(file_path) and file_path.endswith('.csv'):
                        # read zip file
                        file_stem = os.path.splitext(file_name)[0]
                        zip_file_path = os.path.join(case_path, file_stem + self.ZIP_EXTENSION)
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

    def showImageIn3dPlot(self, image, golden_feat, contour=[], predict_feat=[], show_arrow=False, \
        save_figure1=False, index=None):
        fig = plt.figure()
        width = image.shape[1]
        height = image.shape[0]
        # show 2D plot
        ax1 = fig.add_subplot(221)
        ax1.imshow(image, cmap='gray')
        if show_arrow:
            plt.arrow(width / 2.0, height / 2.0, golden_feat[0] * self.DISPLAY_COEFFICIENT, \
                golden_feat[1] * self.DISPLAY_COEFFICIENT, width=self.DISPLAY_ARROW_SIZE)
            if len(predict_feat) > 0:
                plt.arrow(width / 2.0, height / 2.0, predict_feat[0] * self.DISPLAY_COEFFICIENT, \
                    predict_feat[1] * self.DISPLAY_COEFFICIENT, \
                    width=self.DISPLAY_ARROW_SIZE, color='red')
        # show key points
        if len(contour) > 0:
            print('[INFO] Contour points number: ' + str(len(contour)))
            ax1.plot(np.append(contour[:, 0], contour[0, 0]), np.append(contour[:, 1], contour[0, 1]), '-r', lw=3)
        if save_figure1:
            ax1.axis('off')
            file_name = 'figure.png'
            if index:
                file_name = str(index) + '.png'
            fig.savefig(file_name, bbox_inches=ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted()))
        # show 3D plot
        ax2 = fig.add_subplot(222, projection='3d')
        xx, yy = np.meshgrid(np.linspace(0, width, width), np.linspace(0, height, height))
        ax2.quiver(width / 2.0, height / 2.0, self.DISPLAY_OFFSET, golden_feat[0] * self.DISPLAY_COEFFICIENT, \
            golden_feat[1] * self.DISPLAY_COEFFICIENT, golden_feat[2] * self.DISPLAY_COEFFICIENT)
        if len(predict_feat) > 0:
            ax2.quiver(width / 2.0, height / 2.0, self.DISPLAY_OFFSET, predict_feat[0] * self.DISPLAY_COEFFICIENT, \
                predict_feat[1] * self.DISPLAY_COEFFICIENT, predict_feat[2] * self.DISPLAY_COEFFICIENT, color='red')
        ax2.contourf(xx, yy, image, zdir='z', offset=self.DISPLAY_OFFSET, cmap='gray')
        # show chan vese result
        ax3 = fig.add_subplot(223)
        ax3.imshow(chan_vese(image), cmap='gray')
        plt.show()

    def loadImage(self, index=0):
        if index > len(self.image_list):
            print('[ERROR] Index is out of the boundary of the image list.')
            sys.exit(-1)
        zip_file_path = self.image_list[index]['zip']
        image_file_name = self.image_list[index]['name']
        golden_feat = self.image_list[index]['feature']
        with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            print('[INFO] Open image file: ' + image_file_name)
            zip_image = zip_file.read(image_file_name)
            image = cv2.imdecode(np.frombuffer(zip_image, np.uint8), cv2.IMREAD_GRAYSCALE)
            print('[INFO] Image feature: ' + ', '.join([str(x) for x in golden_feat]))
            return image, golden_feat

    def createPositionFeature(self, image, return_contour=False):
        # find contour
        contour_image = chan_vese(image, init_level_set='disk')
        binary = np.asarray(contour_image, dtype='float')
        contour_image = cv2.resize(binary, interpolation=cv2.INTER_AREA, \
            dsize=(int(contour_image.shape[1] / self.POS_FEATURE_DOWNSIZE), \
                int(contour_image.shape[0] / self.POS_FEATURE_DOWNSIZE)))
        pos_feat = contour_image.flatten()
        print('[INFO] Position feature dimenstion: ' + str(np.shape(pos_feat)))
        if return_contour:
            contours, _= cv2.findContours(np.asarray(binary, dtype='uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = np.reshape(max(contours, key=cv2.contourArea), (-1, 2))
            return pos_feat, largest_contour
        return pos_feat, []

    def createHogFeature(self, image):
        hog_result = hog(image, feature_vector=True, pixels_per_cell=(self.HOG_PIXEL_WINDOW, self.HOG_PIXEL_WINDOW), \
            cells_per_block=(1, 1))
        print('[INFO] HOG feature dimenstion: ' + str(np.shape(hog_result)))
        return hog_result

    def readFeatureFile(self, file, sample_limit=-1):
        print('[INFO] Load feature file...')
        feat = []
        golden = []
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file)
            count = 0
            for row in csv_reader:
                feat.append([float(x) for x in row[0].split(',')])
                golden.append([float(x) for x in row[1].split(',')])
                count += 1
                if sample_limit > 0 and count >= sample_limit:
                    break
        print('[INFO] ' + str(count) + ' rows are read')
        return np.array(feat), np.array(golden)

    def trainAndTest(self, train_file, test_file):
        TRAIN_CLUSTER_SIZE = 144 * 6
        TEST_CLUSTER_SIZE = 8 * 6
        CASE_NUM = 520
        print('[INFO] Train and test the feature...')
        with open(train_file) as csv_file_train, open(test_file) as csv_file_test:
            csv_reader_train = csv.reader(csv_file_train)
            csv_reader_test = csv.reader(csv_file_test)
            diff = np.array([])
            for i in range(0, CASE_NUM):
                print('[INFO] Processing case ' + str(i + 1) + '/' + str(CASE_NUM) + '...')
                train_feat = []
                train_golden = []
                count = 0
                for row in csv_reader_train:
                    if count >= TRAIN_CLUSTER_SIZE:
                        break
                    train_feat.append([float(x) for x in row[0].split(',')])
                    # train_feat.append([float(x) for x in row[0].split(',')][0: 28])
                    # train_feat.append([float(x) for x in row[0].split(',')][28: -1])
                    train_golden.append([float(x) for x in row[1].split(',')])
                    count += 1
                test_feat = []
                test_golden = []
                count = 0
                for row in csv_reader_test:
                    if count >= TEST_CLUSTER_SIZE:
                        break
                    test_feat.append([float(x) for x in row[0].split(',')])
                    # test_feat.append([float(x) for x in row[0].split(',')][0: 28])
                    # test_feat.append([float(x) for x in row[0].split(',')][28: -1])
                    test_golden.append([float(x) for x in row[1].split(',')])
                    count += 1
                # train model
                clf = MultiOutputRegressor(svm.SVR())
                # sc_x = StandardScaler()
                # sc_y = StandardScaler()
                # Scale x and y
                # x = sc_x.fit_transform(train_feat)
                # y = sc_y.fit_transform(train_golden)
                # clf.fit(x, y)
                clf.fit(train_feat, train_golden)
                # test model
                # x = sc_x.fit_transform(test_feat)
                # result = sc_y.inverse_transform(clf.predict(x))
                result = clf.predict(test_feat)
                product = [np.clip(np.dot(result[i] / np.linalg.norm(result[i]), \
                    test_golden[i] / np.linalg.norm(test_golden[i])), \
                    -1.0, 1.0) for i in range(0, len(result))]
                diff = np.append(diff, np.rad2deg(np.arccos(product)))
            print('[INFO] The prediction error (angle) is ' + str(np.average(diff)) + '(' + str(np.std(diff)) + ')')
            # other information
            # TRAIN_MAX_LINE = 460800
            # TEST_MAX_LINE = 25600
            # POS_FEAT_DIM = 28
            # HOG_FEAT_DIM = 54
            # ALL: The prediction error (angle) is 15.244228263459785(11.286796067850444)
            #    16.41812944857674(11.118308648586126)
            # CONTOUR: The prediction error (angle) is 17.705939503166988(13.112084351717627)
            #    18.05251460033637(12.496220560725279)
            # HOG: The prediction error (angle) is 16.80747035155666(11.283064643228085)
            #    17.91005449202722(11.547989233324921) (fit)

    def train(self, train_file, sample_limit=-1):
        feat, golden = self.readFeatureFile(train_file, sample_limit)
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

    def testSingleCase(self, index=0):
        image, golden_feat = self.loadImage(index)
        # downsample image
        pos_feat, contour = self.createPositionFeature(image, True)
        hog_feat = self.createHogFeature(image)
        current_feat = np.append(pos_feat, hog_feat)
        with open(self.MODEL_PATH, 'rb') as model_file:
            clf = pickle.load(model_file)
        result = clf.predict(current_feat.reshape(1, -1))
        return image, golden_feat, contour, result

    def test(self, test_file):
        feat, golden = self.readFeatureFile(test_file)
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
        print('[INFO] The prediction error (angle) is ' + str(np.average(diff)) + '(' + str(np.std(diff)) + ')')
        # show result distribution
        # sorted_diff = np.sort(diff)
        # print('[INFO] The best 12800 prediction error (angle) is ' + str(np.average(sorted_diff[0: 12800])) + \
        #     '(' + str(np.std(sorted_diff[0: 12800])) + ')')
        # print('[INFO] The worst 2000 prediction error (angle) is ' + str(np.average(sorted_diff[-2000:])) + \
        #     '(' + str(np.std(sorted_diff[-2000:])) + ')')
        # plt.hist(diff, bins=20)
        # plt.show()
        return np.argsort(diff), result

    def extractFeatureToFile(self, file_path, max_sample_num=-1):
        # generate feature csv
        with open(FEATURE_PATH, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            list_len = len(self.image_list)
            if max_sample_num > 0:
                list_len = max_sample_num
            for index in range(0, list_len):
                print('[INFO] Progress: ' + str(index + 1) + '/' + str(list_len))
                image, golden_feat = self.loadImage(index)
                # downsample image
                pos_feat, _ = self.createPositionFeature(image)
                hog_feat = self.createHogFeature(image)
                current_feat = np.append(pos_feat, hog_feat)
                writer.writerow([','.join([str(x) for x in current_feat]), ','.join([str(x) for x in golden_feat[0: 3]])])

    # constant variables
    # database setting
    IMAGE_FILE_FORMAT = '{:08d}'
    IMAGE_EXTENSION = '.bmp'
    ZIP_EXTENSION = '.zip'
    # display setting
    DISPLAY_OFFSET = 100
    DISPLAY_COEFFICIENT = 100
    DISPLAY_ARROW_SIZE = 3
    # feature setting
    POS_FEATURE_DOWNSIZE = 8
    HOG_PIXEL_WINDOW = 16
    # SVM parameter
    MODEL_PATH = 'model.pkl'
    # private variables
    image_list = []

if __name__ == '__main__':
    DATABASE_PATH = '../s00-09'
    FEATURE_PATH = 'feature.csv'
    LOAD_FEATURE_PATH_TEST = '../s00-09/test_feature5.csv'
    LOAD_FEATURE_PATH_TRAIN = '../s00-09/synth_feature5.csv'
    MODE = DataBaseType.VISUALIZATION_MODE
    database = DataBase(DATABASE_PATH)
    # execute feature extraction or train/test
    if MODE == DataBaseType.EXTRACT_FEAT_MODE:
        database.extractFeatureToFile(FEATURE_PATH, max_sample_num=10)
    elif MODE == DataBaseType.TRAIN_TEST_MODE:
        database.trainAndTest(LOAD_FEATURE_PATH_TRAIN, LOAD_FEATURE_PATH_TEST)
    elif MODE == DataBaseType.TRAIN_ONLY_MODE:
        database.train(LOAD_FEATURE_PATH_TRAIN, sample_limit=144 * 6)
    elif MODE == DataBaseType.TEST_ONLY_MODE:
        database.test(LOAD_FEATURE_PATH_TEST)
    elif MODE == DataBaseType.VISUALIZATION_MODE:
        image_index = 33
        image, golden_feat, contour, predict_feat = database.testSingleCase(image_index)
        database.showImageIn3dPlot(image, golden_feat, contour, predict_feat[0], save_figure1=True, index=image_index)
    sys.exit()
