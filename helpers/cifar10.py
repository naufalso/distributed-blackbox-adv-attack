import tensorflow as tf
from configure.server_configuration import MODEL_SERVER_URL
from helpers.dataset_helper import Dataset_Helper

class Cifar10_Helpers(Dataset_Helper):
    '''
    A Helper class for cifar10 dataset.
    Inherit Dataset_Helper Class

    Parameters:
    model_server_url: str (default: MODEL_SERVER_URL + /cifar-10/predict)
        Remote model server url

    '''

    def __init__(self, model_server_url=None):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        cifar10_test = x_test.astype('float32') / 255
        cifar10_label_test = y_test

        cifar10_input_shape = (32, 32, 3)
        cifar_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

        if model_server_url == None:
            model_server_url = MODEL_SERVER_URL + '/cifar-10/predict'

        Dataset_Helper.__init__(self, cifar10_test, cifar10_label_test, cifar10_input_shape, cifar_class_names, model_server_url)

    


