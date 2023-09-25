import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    # architecture
    inputs = tf.keras.Input(shape=(10, 10))
    outputs = tf.keras.layers.Dense(1)(inputs)
    # model
    model = tf.keras.Model(inputs, outputs)
    # test
    image = np.ones((1, 10, 10))
    print(image)
    print(model.predict(image))
