from flask import jsonify, request, Flask
import numpy as np
import json
from utils.base64_util import Base64_Utils

# TODO: Refactor and Reorganize Code

app = Flask(__name__)
global_best = np.finfo(np.float32).max.astype('float')
global_best_pos = np.zeros(0)
base64_util = Base64_Utils()

@app.route('/global-best', methods=['POST'])
def global_best_update():
    global global_best, global_best_pos, base64_util
    
    json_data = request.get_json(force=True)
    if not json_data:
        return {'sucess': False, 'message': 'No input data provided'}, 400

    if json_data['global_best'] < global_best:
        global_best =  json_data['global_best']
        global_best_pos = base64_util.decode_numpy(json_data['global_best_pos'])
        print('Updating Global Best', global_best, global_best_pos.shape)

    data = json.dumps({"global_best": global_best, 'global_best_pos': base64_util.encode_numpy(global_best_pos)})
    return data, 200

@app.route('/global-best/reset', methods=['POST'])
def global_best_reset():
    global global_best, global_best_pos, base64_util
    
    global_best = np.finfo(np.float32).max.astype('float')
    global_best_pos =  np.zeros(0)
    
    data = json.dumps({"global_best": global_best, 'global_best_pos': base64_util.encode_numpy(global_best_pos)})
    return data, 200

@app.route('/global-best', methods=['GET'])
def global_best_get():
    global global_best, global_best_pos
    data = json.dumps({"global_best": global_best, 'global_best_pos':  base64_util.encode_numpy(global_best_pos)})
    print(global_best, global_best_pos.shape)
    return data, 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000, debug=False)