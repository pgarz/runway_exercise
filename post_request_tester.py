
import requests
import base64
import json

url = 'http://0.0.0.0:8000/generate'

image_path = './blurred_sharp/blurred/248.png'
# image_path = './edge_case_images/macbook.png'

with open(image_path, 'rb') as fd:
    b64data = base64.b64encode(fd.read())

json_data = "data:image/jpeg;base64," + b64data.decode("utf-8")
data = {'blurred_image': json_data}
headers = {'Content-type': 'application/json'}
requests.post(url, data=json.dumps(data), headers=headers)
