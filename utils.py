import os
from sklearn.utils import shuffle
from time import ctime
import scipy.io as sio
import numpy as np
from scipy.misc import imread, imresize, imsave

# UCF101 frame shape (240, 320, 3)#

HEIGHT = 128
WIDTH = 171
FRAMES = 16
RE_SIZE = 240 
CROP_SIZE = 112 
CHANNELS = 3
BATCH_SIZE = 16 

num_class=24

frame_dir = '/home/ccho/Documents/Data/UCF101/frames/'

#####################################
# Video Processing Utilities
# Written by Quang-Vinh Tran
# Date Created: Nov 13th, 2017
# Date Modified: Nov 13th, 2017
#####################################

def clipread(paths, offsets, size=(240, 240), crop_size=(224, 224), mode='RGB', interp='bilinear'):
    """
    Read video clip, resize to height and width resolution, crop the clip, then resize to crop height and crop width
    :param paths: Paths to  N (N = 16 for C3D) consecutive frames
    :param offsets: Crop window offset in form of [from_H, to_H, from_W, to_W], example: (0, 112, 24, 136)
    :param size: Tuple, size of the output image
    :param crop_size: Tuple, size of the output cropped image
    :param mode: 'RGB' or 'L' for gray scale
    :param interp: Interpolation to use for re-sizing, example: 'nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic'
    :return: Cropped clip (depth, crop_height, crop_width, channels) in float32 format, pixel values in [0, 255]
    """
    assert mode in ('RGB', 'L'), 'Mode is either RGB or L'

    clips = []
    for file_name in paths:
        # Read video frame
        im = imread(file_name, mode=mode)

        # Resize frame to init resolution and crop then resize to target resolution
        #im = imresize(im, size=crop_size, interp=interp)
         
        if mode == 'RGB':
            im = imresize(im, size=size, interp=interp)
            data = im[offsets[0]:offsets[1], offsets[2]:offsets[3], :]
            im = imresize(data, size=crop_size, interp=interp)
        else:
            im = imresize(im, size=size, interp=interp)
            data = im[offsets[0]:offsets[1], offsets[2]:offsets[3]]
            im = imresize(data, size=crop_size, interp=interp)
        
        clips.append(im)

    clips = np.array(clips, dtype=np.float32)

    if mode == 'RGB':
        return clips
    return np.expand_dims(clips, axis=3)


def randcrop(scales, size=(240, 240)):
    """
    Generate random offset for crop window
    :param scales: List of scales for crop window, example: (128, 112, 96, 84)
    :param size: Tuple, size of the image
    :return: Crop window offsets in form of (from_H, to_H, from_W, to_W), example: (0, 112, 24, 136)
    """
    scales = np.array(scales) if isinstance(scales, (list, tuple)) else np.array([scales])
    scale = scales[np.random.randint(len(scales))]
    height, width = size

    max_h = height - scale
    max_w = width - scale

    off_h = np.random.randint(max_h) if max_h > 0 else 0
    off_w = np.random.randint(max_w) if max_w > 0 else 0

    return off_h, off_h + scale, off_w, off_w + scale


def centercrop(scale, size=(128, 171)):
    """
    Generate center offset for crop window
    :param scale: Int, a scale for crop window, example: 112
    :param size: Tuple, size of the image, example: (128, 171)
    :return: Crop window offsets in form of (from_H, to_H, from_W, to_W), example: (8, 120, 29, 141)
    """
    height, width = size

    off_h = np.ceil((height - scale) / 2).astype(int)
    off_w = np.ceil((width - scale) / 2).astype(int)

    return off_h, off_h + scale, off_w, off_w + scale

def read_train(tr_file):
    path, _, cls = tr_file.split(' ')
    frm = len([f for f in os.listdir(os.path.join(frame_dir, path)) if '.jpg' in f])
    start = np.random.randint(int(frm) - FRAMES)


    v_paths = [frame_dir + path + 'frm_%06d.jpg' % (f + 1) for f in range(start, start + FRAMES)]
    
    offsets = randcrop(scales=[128, 112, 96, 84], size=(RE_SIZE, RE_SIZE))
    voxel = clipread(v_paths, offsets, size=(RE_SIZE, RE_SIZE), crop_size=(CROP_SIZE, CROP_SIZE), mode='RGB')

    is_flip = np.random.rand(1, 1).squeeze() > 0.5
    if is_flip:
        voxel = np.flip(voxel, axis=2)

    return voxel, np.float32(cls)

def read_test(tst_file):
    path, start, cls = tst_file.split(' ')
    start = int(start)

    v_paths = [frame_dir + path + 'frm_%06d.jpg' % (f + 1) for f in range(start, start + FRAMES)]
    offsets = [0, CROP_SIZE, 0, CROP_SIZE] 
    #offsets = randcrop(scales=[128, 112, 96, 84], size=(CROP_SIZE, CROP_SIZE))
    voxel = clipread(v_paths, offsets, size=(CROP_SIZE, CROP_SIZE), crop_size=(CROP_SIZE, CROP_SIZE), mode='RGB')

    return voxel, np.float32(cls)
