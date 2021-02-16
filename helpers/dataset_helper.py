import numpy as np
import json
import requests
import time

from PIL import Image
from configure.server_configuration import MODEL_SERVER_URL
from utils.base64_util import Base64_Utils



class Dataset_Helper:
    '''
    A Helper Class for Dataset in Numpy Format
    This class contains methods for querying the dataset into remote model server

    Attributes
    ----------
    datasets: numpy
        datasets (images) in numpy array used for testing
    labels: numpy
        expected labels indexes of the dataset in numpy array
    shape: tuple
        datasets shape
    label_names: list(string) 
        list of label name in string
    model_server_url: String
        remote model server endpoint for query/predict the dataset
    '''

    def __init__(self, datasets, labels, shape, label_names, model_server_url):
        '''
        Parameters
        ----------
        datasets: numpy
            datasets (images) in numpy array used for testing
        labels: numpy
            expected labels indexes of the dataset in numpy array
        shape: tuple
            datasets shape
        label_names: list(str) 
            list of label name in string
        model_server_url: str
            remote model server endpoint for query/predict the dataset
        '''

        # Public Attributes
        self.datasets = datasets
        self.labels = labels
        self.shape = shape
        self.label_names = label_names
        self.model_server_url = model_server_url

        single_batch_shape = list(self.shape)
        single_batch_shape[:0] = [1]
        self.single_batch_shape = tuple(single_batch_shape)

        flattened_shape = np.prod(single_batch_shape)
        self.flattened_shape = tuple([flattened_shape])

        # Private Attributes
        self.__base64_utils = Base64_Utils()
        self.__json_headers = {"content-type": "application/json"}

    def get_batch_shape(self, batch_size):
        multiple_batch_shape = list(self.shape)
        multiple_batch_shape[:0] = [batch_size]
        return multiple_batch_shape

    def load_img(self, file_name):
        '''Load image from local directory into numpy array

        Parameters
        ----------
        file_name: str
            image file directory

        Returns
        -------
        image: numpy
            image in numpy array
        '''

        img = Image.open(file_name)
        img_arr = np.array(img) / 255.0
        img_arr = img_arr.astype(np.float32)
        return img_arr[:, :, 0]

    def softmax(self, probs):
        '''Normalize the probability distribution using softmax function

        Parameters
        ----------
        probs: numpy
            List of probability that will be normalized

        Returns
        -------
        normalized_probs: numpy
            Probability normalized by softmax function
        '''

        exp_data = np.exp(probs)
        sum_exp_data = np.sum(exp_data)
        normalized_probs = exp_data / sum_exp_data
        return normalized_probs

    def query(self, imgs, with_softmax=False):
        '''Query images data into remote model server

        Parameters
        ----------
        imgs: numpy
            List of image that will be queried into remote model server
        with_softmax: bool (default False)
            enable this to normalized the probability with softmax

        Returns
        -------
        predictions: numpy
            List of probability of predicted images
        '''

        while True:
            try:
                base64_imgs = self.__base64_utils.encode_numpy(imgs)
                data = json.dumps({"data": base64_imgs})
                json_response = requests.post(
                    self.model_server_url, data=data, headers=self.__json_headers)
                predictions = json.loads(json_response.text)['predictions']

                if with_softmax:
                    return self.softmax(predictions)
                else:
                    return predictions
            except:
                print('query error, retrying')
                time.sleep(0.5)
                

    def predict(self, img, with_softmax=False):
        '''Predict a single image

        Parameters
        ----------
        img: numpy
            an image that will be predicted into remote server
        with_softmax: bool (default False)
            enable this to normalized the probability with softmax


        Returns
        -------
        predicted_label_with_prob: str
            predicted label with probability/confidence score
        predicted_index: int
            predicted class index
        predicted_prob: numpy float
            predicted probabiltiy/confidence score
        '''
        result = self.query(img.reshape(self.single_batch_shape), with_softmax)

        predicted_index = np.argmax(result).astype('int')
        predicted_prob = result[0][predicted_index]
        predicted_label = self.label_names[predicted_index]
        predicted_label_with_prob = '{} ({:.0f} %)'.format(
            predicted_label, predicted_prob * 100.0)

        return predicted_label_with_prob, predicted_index, predicted_prob

    def calculate_distance(self, original_img, poisoned_img):
        ''' Calculate L_p norm distance between two images

        Parameters
        ----------
        original_img: numpy
            original image that has been flattened
        posionded_img: numpy
            poision image that has been flattened. Must have same shape with original_img

        Returns
        -------
        l0: float
            L-0 distance
        l2: float
            L-2 distance
        linf: float
            L-Infinity distance
        '''

        difference = original_img-poisoned_img

        l0 = np.linalg.norm(difference, ord=1)
        l2 = np.linalg.norm(difference, ord=2)
        linf = np.linalg.norm(difference, ord=np.inf)

        return l0, l2, linf
