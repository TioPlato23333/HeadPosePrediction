import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    # architecture
    inputs = tf.keras.Input(shape=(10, 10, 1))
    outputs = tf.keras.layers.Conv2D(1, 3, input_shape=(10, 10, 1))(inputs)
    # outputs = tf.keras.layers.Dense(1)(inputs)
    # model
    model = tf.keras.Model(inputs, outputs)
    # test
    image = np.ones((1, 10, 10, 1))
    for layer in model.layers:
        print(layer.output)
    # print(model.predict(image))
