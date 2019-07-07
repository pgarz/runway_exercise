from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import torch
# Extra stuff needed for image processing
import torchvision.transforms as transforms
from PIL import Image


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def __init__(self, opt):
        assert(not opt.isTrain)
        super(TestModel, self).__init__(opt)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, False,
                                      opt.learn_residual)
        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)


        if opt.use_online_data:
            self.using_online_data = True
            self.single_image_transform = self.get_transform(opt)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        temp = self.input_A.clone()
        temp.resize_(input_A.size()).copy_(input_A)
        self.input_A = temp
        self.image_paths = input['A_paths']


    # special function that sets the input directly in this way
    def set_online_input(self, input_img_tensor):
        assert(self.using_online_data)
        # This input is actually a PIL Image and needs to be modified
        # This call does all the normalization stuff

        # Add some padding on the fly if the image has odd dimensions
        # extra_pad = transforms.Pad((1,1,0,0))
        # input_img_tensor = extra_pad(input_img_tensor)
        #TODO: also write some code here so that we add padding until the image is a square
        # Could also just crop to an even size as well if that's easier.
        print("set_online_input:")
        print(input_img_tensor)

        height = input_img_tensor.size[-2]
        width = input_img_tensor.size[-1]

        if height % 2 != 0 or width % 2 != 0 or height != width:
            print("Adding a padding of 1 to the image")
            print(input_img_tensor.size)

            # Code for padding
            # self.single_image_transform = [transforms.Pad((1,1,0,0))] + [self.single_image_transform]
            # self.single_image_transform = transforms.Compose(self.single_image_transform)

            # Code for instead applying an image crop
            min_dim = min([height, width])
            # fit to nearest even dimension
            min_dim = min_dim -1 if min_dim % 2 != 0 else min_dim


            self.single_image_transform = [transforms.CenterCrop((min_dim, min_dim))] + [self.single_image_transform]
            self.single_image_transform = transforms.Compose(self.single_image_transform)


        input_img_tensor = self.single_image_transform(input_img_tensor)

        input_img_tensor = input_img_tensor.unsqueeze(0)
        input_A = input_img_tensor
        temp = self.input_A.clone()
        temp.resize_(input_A.size()).copy_(input_A)
        self.input_A = temp

        # manually set image_paths to be empty list
        self.image_paths = []


    def test(self):
        with torch.no_grad():
            self.real_A = Variable(self.input_A)

            self.fake_B = None
            self.fake_B = self.netG.forward(self.real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

    def get_transform(self, opt):
        transform_list = []

        # osize = [opt.loadSizeX, opt.loadSizeY]
        # transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)


