
import requests
import base64
import json

# This file was made as a test script to locally send images in an easy way with one click. Easier to use than the
# command-line based instructions in the tutorial.

url = 'http://0.0.0.0:8000/generate'

# image_path = './blurred_sharp/blurred/248.png'
# image_path = './edge_case_images/macbook.png'
image_path = './edge_case_images/train.jpg'

with open(image_path, 'rb') as fd:
    b64data = base64.b64encode(fd.read())

json_data = "data:image/jpeg;base64," + b64data.decode("utf-8")
data = {'blurred_image': json_data}
headers = {'Content-type': 'application/json'}
requests.post(url, data=json.dumps(data), headers=headers)
