import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    # architecture
    inputs = tf.keras.Input(shape=(10, 10, 1))
    # outputs = tf.keras.layers.Conv2D(1, 3, input_shape=(10, 10, 1))(inputs)
    # outputs = tf.keras.layers.Dense(1)(inputs)
    outputs = tf.keras.layers.MaxPooling2D()(inputs)
    # model
    model = tf.keras.Model(inputs, outputs)
    print('[INFO] Layer architecture:')
    for layer in model.layers:
        print(layer.output)
    # test
    # image = np.ones((1, 10, 10, 1))
    image = tf.random.uniform((1, 10, 10, 1))
    print('[INFO] Input/output:')
    print(image)
    print(model.predict(image))
