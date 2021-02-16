import tensorflow as tf
from configure.server_configuration import MODEL_SERVER_URL
from helpers.dataset_helper import Dataset_Helper

class Mnist_Helpers(Dataset_Helper):
    '''
    A Helper class for mnist dataset.
    Inherit Dataset_Helper Class

    Parameters:
    model_server_url: str (default: MODEL_SERVER_URL + /mnist/predict)
        Remote model server url

    '''
    def __init__(self, model_server_url=None):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        mnist_test = x_test.astype('float32') / 255
        mnist_label_test = y_test

        mnist_input_shape = (28, 28, 1)
        mnist_class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']

        if model_server_url == None:
            model_server_url = MODEL_SERVER_URL + '/mnist/predict'

        Dataset_Helper.__init__(self, mnist_test, mnist_label_test, mnist_input_shape, mnist_class_names, model_server_url)
