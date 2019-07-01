import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
from ssim import SSIM
from PIL import Image

# Stuff from the Runway tutorial
import runway
from runway.data_types import category, vector, image, number, text

# Added for some command-line trickery
import sys


# The setup() function runs once when the model is initialized, and will run
# again for each well formed HTTP POST request to http://localhost:8000/setup.
@runway.setup(options={ 'gpu_ids': number(default=-1)
                       , 'model': text(default="test")
                       , 'dataset_mode': text(default="single")
                       # , 'learn_residual': text(default="") #TODO: double check that this is okay
                       })
def setup(opts):
    # Add some argumetns that we'll always tend to use for basic inference
    gpu_id = str(opts['gpu_ids'])
    model_type = opts['model']
    dataset_mode = opts['dataset_mode']


    default_args = ['--gpu_ids', gpu_id, '--dataroot',
     './blurred_sharp/blurred', '--model', model_type, '--dataset_mode', dataset_mode, '--learn_residual',
    '--use_online_data']

    sys.argv += default_args

    # Actually parse the arguments
    opt = TestOptions().parse()

    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip


    # data_loader = CreateDataLoader(opt)
    # dataset = data_loader.load_data()


    model = create_model(opt)


    return model

inputs = { 'blurred_image': image() }
outputs = { 'sharpened_image': image(width=256, height=256) }

# The @runway.command() decorator is used to create interfaces to call functions
# remotely via an HTTP endpoint. This lets you send data to, or get data from,
# your model. Each command creates an HTTP route that the Runway app will use
# to communicate with your model (e.g. POST /generate). Multiple commands
# can be defined for the same model.
@runway.command('generate', inputs=inputs, outputs=outputs)
def generate(model, input_args):
    # Functions wrapped by @runway.command() receive two arguments:
    # 1. Whatever is returned by a function wrapped by @runway.setup(),
    #    usually a model.
    # 2. The input arguments sent by the remote caller via HTTP. These values
    #    match the schema defined by inputs.

    print("IN GENERATE")
    print(input_args)

    input_img = input_args['blurred_image']

    model.set_online_input(input_img)
    model.test()
    visuals = model.get_current_visuals()

    # convert this to a pytorch tensor

    # Apply normalizing transformations
    return visuals


# The runway.run() function triggers a call to the function wrapped by
# @runway.setup() passing model_options as its single argument. It also
# creates an HTTP server that listens for and fulfills remote requests that
# trigger commands.
if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8000, model_options={ 'gpu_ids': -1, 'model': 'test', 'dataset_mode': 'single' })