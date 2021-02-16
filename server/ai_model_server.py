from flask import jsonify, request, Flask
import numpy as np
import tensorflow as tf
import json
from utils.base64_util import Base64_Utils

# TODO: Refactor and reorganize code
# TODO: Enable CW and own model

app = Flask(__name__)

def fn(correct, predicted):
    train_temp = 1
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted/train_temp)

def softmax(data):
    exp_data = np.exp(data)
    sum_exp_data = np.sum(exp_data)
    y = exp_data / sum_exp_data
    return y

mnist_model_cw = tf.keras.models.load_model('model/mnist_cw.h5', custom_objects={'fn': fn})
cifar10_model_cw = tf.keras.models.load_model('model/cifar_cw.h5', custom_objects={'fn': fn})

mnist_model = tf.keras.models.load_model('model/mnist.h5')
cifar10_model = tf.keras.models.load_model('model/cifar_10.h5')
base64_util = Base64_Utils()

# Own Trained Model #

@app.route('/mnist/predict', methods=['POST'])
def mnist_predict():
    json_data = request.get_json(force=True)
    if not json_data:
        return {'sucess': False, 'message': 'No input data provided'}, 400
    data =  json_data['data']
    img_data = base64_util.decode_numpy(data)
    img_data = img_data.reshape((int(img_data.shape[0] / 784), 28, 28, 1))
    result = mnist_model.predict(img_data)
    data = json.dumps({'sucess': True, "predictions": result.tolist()})
    return data, 200

@app.route('/cifar-10/predict', methods=['POST'])
def cifar10_predict():
    json_data = request.get_json(force=True)
    if not json_data:
        return {'sucess': False, 'message': 'No input data provided'}, 400
    data =  json_data['data']
    img_data = base64_util.decode_numpy(data)
    img_data = img_data.reshape((int(img_data.shape[0] / 3072), 32, 32, 3))
    results = cifar10_model.predict(img_data)
    data = json.dumps({'sucess': True, "predictions": results.tolist()})
    return data, 200

# CW Trained Model #

@app.route('/mnist-cw/predict', methods=['POST'])
def mnist_cw_predict():
    json_data = request.get_json(force=True)
    if not json_data:
        return {'sucess': False, 'message': 'No input data provided'}, 400
    data =  json_data['data']
    img_data = base64_util.decode_numpy(data)
    img_data = img_data.reshape((int(img_data.shape[0] / 784), 28, 28, 1))
    result = mnist_model_cw.predict(img_data)
    data = json.dumps({'sucess': True, "predictions": result.tolist()})
    return data, 200

@app.route('/cifar-10-cw/predict', methods=['POST'])
def cifar10_cw_predict():
    json_data = request.get_json(force=True)
    if not json_data:
        return {'sucess': False, 'message': 'No input data provided'}, 400
    data =  json_data['data']
    img_data = base64_util.decode_numpy(data)
    img_data = img_data.reshape((int(img_data.shape[0] / 3072), 32, 32, 3))
    results = cifar10_model_cw.predict(img_data)
    data = json.dumps({'sucess': True, "predictions": results.tolist()})
    return data, 200