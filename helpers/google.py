import numpy as np
from PIL import Image
from utils.base64_util import Base64_Utils
from io import BytesIO
import base64
import requests
import json
import configure.constant as constant

class Google_Vision_Helpers:
    def __init__(self):
        super().__init__()
        self.base64_utils = Base64_Utils()
        self.json_header = {"content-type": "application/json"}
        self.api_key = constant.GOOGLE_API_KEY
        self.base_url ='https://vision.googleapis.com/v1/'
        self.query_img_url = self.base_url + 'images:annotate?key=' + self.api_key

    def load_img(self, file_name):
        img = Image.open(file_name)
        img_arr = np.array(img) / 255.0
        return img_arr.astype(np.float32)
    
    def encode_img(self, image):
        reverted_img = (image * 255.0).astype('uint8')
        img_obj = Image.fromarray(reverted_img)
        buffered = BytesIO()
        img_obj.save(buffered, format='jpeg')
        base64_img = str(base64.b64encode(buffered.getvalue()))[2:-1]
        return base64_img

    def query_img(self, image):
        base64_img = self.encode_img(image)
        data = json.dumps(
            {
                "requests": [
                    {
                    "image": {
                        "content": base64_img
                    },
                    "features": [
                        {
                        "maxResults": 20,
                        "type": "LABEL_DETECTION"
                        }
                    ]
                    }
                ]
            }
        )
        result = requests.post(self.query_img_url, data=data, headers=self.json_header)
        result_obj = json.loads(result.text)
        # print(result.text)
        # print(result_obj)
        return result_obj['responses'][0]['labelAnnotations']

    def get_score(self, scores, label_name):
        # print(scores)
        confidence_score = 0.0
        highest_score = False
        i = 0
        for score in scores:
            if score['description'].lower() == label_name.lower():
                confidence_score = score['score']
                if i == 0:
                    highest_score = True
            i+=1
        return confidence_score, highest_score
    
    def get_total_score(self, scores, label_names):
        # print(scores)
        confidence_scores = np.zeros(len(label_names))
        # confidence_score = 0.0
        for score in scores:
            for i in range(len(label_names)):
                if score['description'].lower() == label_names[i].lower():
                    confidence_scores[i] = score['score']
        
        return np.sum(confidence_scores)

    def calculate_distance(self, original_img, poisoned_img):
        l0 = np.linalg.norm(original_img-poisoned_img, ord=1)
        l2 = np.linalg.norm(original_img-poisoned_img, ord=2)
        linf = np.linalg.norm(original_img-poisoned_img, ord=np.inf)
        # print('l0', l0)
        # print('l2', l2)
        # print('linf', linf)
        return l0, l2, linf