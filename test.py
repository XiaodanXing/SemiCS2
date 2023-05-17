import os
from collections import OrderedDict
import numpy as np
import cv2
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import copy
from torchmetrics.functional import dice_score


os.environ["CUDA_VISIBLE_DEVICES"] =  "3"
opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# Configure dataloaders
from data.biDataset import biDatasetCTStable
from torch.utils.data import DataLoader

test_data = biDatasetCTStable(2000,3000,tr=opt.tr
                              ,num_class=3,path='/media/NAS02/DataCam/CycleGANdataset/USegv5',
                              )

test_loader = DataLoader(test_data , batch_size=1, shuffle=True)


visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))



# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx


for i, data in enumerate(test_loader):
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()

    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()

    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)

        exit(0)
    minibatch = 1 
    if opt.engine:

        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    else:        
        generated = model.module.inference(label=Variable(data['instance']),
            inst=Variable(data['instance']))


    visuals = OrderedDict([('input_label', util.tensor2im(data['instance'][0,0,:,:],
                                                                  )),
                           ('real_image', util.tensor2im(data['image'][0, 0, :, :])),
                                   ('synthesized_image', util.tensor2im(generated[0].data[0,0,:,:])),
                           ('synthesized_segmentation',
        util.tensor2im(torch.argmax(generated[1].data[0, :, :, :], dim=0),
                       label_img=opt.output_numclass,coloring=False)),
        ('real_segmentation',
    util.tensor2im(torch.argmax(data['image'][0, 1:, :, :], dim=0), label_img=opt.output_numclass,
                                                                  coloring=False))
                            ])



    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()

