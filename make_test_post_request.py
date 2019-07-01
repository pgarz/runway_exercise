
import requests
import base64

url = 'http://0.0.0.0:8000/generate'

image_path = './blurred_sharp/blurred/248.png'

with open(image_path, 'rb') as fd:
    b64data = base64.b64encode(fd.read())

files = {'blurred_image': ('248.png', b64data, 'application/json')}
headers = {'Content-type': 'application/json'}
requests.post(url, files=files, headers=headers)
