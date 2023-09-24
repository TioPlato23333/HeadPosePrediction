import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
from PIL import Image
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array 
from tensorflow.keras.applications import resnet50

def printNetworkSummary2File(model, file):
    original_stdout = sys.stdout
    with open(file, 'w') as f:
        sys.stdout = f
        model.summary()
        sys.stdout = original_stdout

if __name__ == '__main__':
    filename = 'banana.jpg' 
    ## load an image in PIL format 
    original = load_img(filename, target_size=(224, 224)) 
    print('PIL image size', original.size)
    # plt.imshow(original) 
    # plt.show()
    # convert the PIL image to a numpy array 
    numpy_image = img_to_array(original)
    plt.imshow(np.uint8(numpy_image)) 
    print('numpy array size', numpy_image.shape) 
    # Convert the image / images into batch format 
    image_batch = np.expand_dims(numpy_image, axis=0) 
    print('image batch size', image_batch.shape) 
    # prepare the image for the resnet50 model 
    processed_image = resnet50.preprocess_input(image_batch.copy()) 
    # create resnet model 
    # resnet_model = resnet50.ResNet50()
    resnet_model = resnet50.ResNet50(weights='./weights/', include_top=False)
    printNetworkSummary2File(resnet_model, 'network_summary')
    resnet_model.save_weights('./weights/')
    # try to use another architecture
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = resnet_model(inputs, training=False)
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inputs, outputs)
    printNetworkSummary2File(model, 'network_summary_2')
    '''
    # get the predicted probabilities for each class
    predictions = resnet_model.predict(processed_image)
    # convert the probabilities to class labels
    label = decode_predictions(predictions)
    print(label)
    '''
    '''
    # try fit function
    print(processed_image.shape)
    print(predictions.shape)
    resnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives()])
    data = tf.random.uniform([1, 224, 224, 3])
    labels = tf.random.uniform([1, 1000])
    resnet_model.fit(data, labels)
    predictions2 = resnet_model.predict(processed_image)
    # convert the probabilities to class labels
    label2 = decode_predictions(predictions2)
    print(label2)
    '''
    '''
    # extract middle layer output
    feature_extractor = tf.keras.Model(inputs=resnet_model.inputs, \
        outputs=[layer.output for layer in resnet_model.layers])
    features = feature_extractor(processed_image)
    IMG_DIR = 'temp_result/'
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
    for feature in features:
        feat_array = feature.numpy()
        for i in range(feat_array.shape[-1]):
            raw_img = (feat_array[0, ..., i] * 255).astype(np.uint8)
            if raw_img.ndim != 2:
                print('[INFO] Image ' + str(feat_array.shape) + ' can not be visualized.')
                continue
            img = Image.fromarray(raw_img)
            img_name = str(feat_array.shape) + '_' + str(i) + '.jpg'
            img.save(IMG_DIR + img_name)
            print('[INFO] Image ' + img_name + ' saved.')
    '''
