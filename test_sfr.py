# Imports
import sys
caffe_root = '/usr/lib/python3/dist-packages' # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
import pickle
import numpy as np
from PIL import Image
import glob
import json

folder = 'n01443537'
dnn = 'ResNet152'
img_crop = 224
label_path = "imagenet_labels.json"
with open(label_path) as f:
    label_dict = json.load(f)
model_def='../Selective-feature-regeneration/Prototxt/ResNet152/deploy_resnet152_FRU.prototxt'
pretrained_model='../Selective-feature-regeneration/Prototxt/ResNet152/resnet152_FRU.caffemodel'

caffe_root='/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val/' + folder

#dataset
IMAGENET_PATH = '/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/'
#full_val = 50000
#full_index = np.arange(0, full_val)
index_test = np.load(IMAGENET_PATH + '/validation/index_test.npy').astype(np.int64)
#index_train = [x for x in full_index if x not in index_test]

# Create a net object
net = caffe.Net(model_def, pretrained_model, caffe.TEST)

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,         # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 227x227

correct = 0
i = 0
total = 0
for f in glob.iglob(IMAGENET_PATH + "validation/val/*"):
    if i not in index_test:
        i = i + 1
        continue
    i = i + 1

    print('Processing image: {} {}'.format(i, f))
    img = Image.open(f).convert('RGB')
    w, h = img.size

    # image resizing for VGG16, VGG_F and ResNet152 maintains the original aspect ratio of the image
    '''
    if w >= h:
        img = img.resize((256 * w // h, 256))
    else:
        img = img.resize((256, 256 * h // w))
    '''
    img = np.transpose(np.asarray(img), (2, 0, 1))
    img = img[[2, 1, 0], :, :]
    img = img.astype(np.float32)
    img = img[:, (img.shape[1] - img_crop) // 2:(img.shape[1] + img_crop) // 2,
               (img.shape[2] - img_crop) // 2:(img.shape[2] + img_crop) // 2]

    img = img[np.newaxis,:]

    # Mean subtraction values for ResNet152v2 model are different than the other 4 models provided
    if dnn=='ResNet152':
        img[:, 0, :, :] -= 102.98
        img[:, 1, :, :] -= 115.947
        img[:, 2, :, :] -= 122.772
    else:
        img[:, 0, :, :] -= 103.939
        img[:, 1, :, :] -= 116.779
        img[:, 2, :, :] -= 123.68

    net.blobs['data'].reshape(*img.shape)
    net.blobs['data'].data[...] = img

    net.forward()
    pred = net.blobs['prob'].data[0]
    pred_ids = np.argsort(pred)[-5:][::-1]

    print("Top 1 prediction: ",  label_dict[str(pred_ids[0])][1], ", Confidence score: ", str(np.max(pred)))
    print("Top 5 predictions: ", [label_dict[str(pred_ids[k])][1] for k in range(5)])

    correct = correct + (label_dict[str(pred_ids[0])][0] in f)
    total = total + 1

print('Clean sample top 1 accuracy: {}%'.format(correct / total * 100.))