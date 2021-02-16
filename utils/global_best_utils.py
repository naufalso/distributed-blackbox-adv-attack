import configure.server_configuration as server_config
import requests
import json
import base64
import numpy as np
from utils.base64_util import Base64_Utils

class Global_Best_Utils:
    def __init__(self, server_url=None):
        if server_url == None:
            print('SERVER_URL is Undefined. Using default')
            server_url = server_config.GLOBAL_BEST_SERVER_URL
        self.global_best_server_url = server_url
        self.json_header = {"content-type": "application/json"}
        self.base64_utils = Base64_Utils()
    
    def query_global_best(self):
        json_response = requests.get(
            self.global_best_server_url + '/global-best', headers=self.json_header)
        data = json.loads(json_response.text)
        global_best = data['global_best']
        global_pos_best = self.base64_utils.decode_numpy(data['global_best_pos'])
        return global_best, global_pos_best
 
    def update_global_best(self, global_best, global_best_pos):
        if type(global_best).__module__ == np.__name__:
            global_best = global_best.astype('float')
        
        data = json.dumps({"global_best": global_best, 'global_best_pos': self.base64_utils.encode_numpy(global_best_pos)})
        json_response = requests.post(
            self.global_best_server_url + '/global-best', data=data, headers=self.json_header)
        data = json.loads(json_response.text)
        return data['global_best'], self.base64_utils.decode_numpy(data['global_best_pos'])

    def reset_global_best(self):
        json_response = requests.post(
            self.global_best_server_url + '/global-best/reset', headers=self.json_header)
        data = json.loads(json_response.text)
        return data['global_best'], self.base64_utils.decode_numpy(data['global_best_pos'])

if __name__ == "__main__":
    global_best_utils = Global_Best_Utils()
    print(global_best_utils.reset_global_best())
    # TODO add query and update testing
