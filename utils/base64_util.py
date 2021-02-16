import numpy as np
import base64

class Base64_Utils:
    def __init__(self):
        super().__init__()

    def encode_numpy(self, np_data):
        return str(base64.b64encode(np_data.tobytes()), 'UTF-8')

    def decode_numpy(self, base64_data):
        return np.frombuffer(base64.b64decode(base64_data), dtype=np.float32)

if __name__ == "__main__":
    base64_util = Base64_Utils()
    t = np.arange(25, dtype=np.float32)
    print(t)
    np_base64 = base64_util.encode_numpy(t)
    print(np_base64)
    np_ori = base64_util.decode_numpy(np_base64)
    print(np_ori)
