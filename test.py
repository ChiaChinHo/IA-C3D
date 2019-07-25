import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import scipy.io as sio
import numpy as np
from scipy.misc import imresize
import copy
from model import *
from utils import *
import scipy.ndimage.measurements as mnts
from time_smoothing import *
from argparse import ArgumentParser

parser = ArgumentParser(description='Process some integers.')
# the dictionay containing training/test list files 
parser.add_argument('-t', '--testfile', type=str,
                    help='data list')
parser.add_argument('-p', '--path', type=str,
                    help='data path')
parser.add_argument('-n', '--num', type=int, default=24,
                    help='number of classes')
parser.add_argument('-th', '--threshold', type=float, default=0.01,
                    help='threshold')

args = parser.parse_args()

FRAMES = 16
CROP_SIZE = 112 
CHANNELS = 3
BATCH_SIZE = 32

num_class= args.num
frame_dir = args.path
threshold = args.threshold
s_th = 2.0/(num_class+1)


testfilepath = os.path.join(args.testfile, 'test.txt')
with open(testfilepath, 'r') as f:
    lines = f.readlines()
val_files = [line for line in lines if len(line) > 0]



# Define placeholders
x = tf.placeholder(tf.float32, shape=(None, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), name='input_x')
y = tf.placeholder(tf.int64, None, name='input_y')
training = tf.placeholder(tf.bool, name='training')

# Define the C3D model for UCF101
inputs = x - tf.constant([96.6], dtype=tf.float32, shape=[1, 1, 1, 1, 1])
#inputs = tf.scalar_mul(2/255.0, x) - tf.constant([1.0], dtype=tf.float32, shape=[1, 1, 1, 1, 1])
logits, [y_att_1, AC_1, _] = c3d(inputs=inputs, num_class=num_class, training=training)
labels = tf.one_hot(y, num_class, name='labels')

def acc_func(logits, labels=labels, y=y):
    if len(logits.get_shape().as_list()) > 2:
        logits = tf.reduce_mean(tf.nn.softmax(logits), axis=1)

    correct_opt = tf.equal(tf.argmax(logits, -1), y, name='correct')
    return tf.reduce_mean(tf.cast(correct_opt, tf.float32), name='accuracy')

# Main net operations
with tf.variable_scope('main_net'):
    acc_opt = acc_func(logits)

with tf.variable_scope('subnet_1'):
    acc_subnet_att_1 = acc_func(y_att_1)

# Final operation
with tf.variable_scope('final'):
    y_final = tf.scalar_mul(0.5, tf.nn.softmax(logits)) + tf.scalar_mul(0.5, tf.reduce_mean(tf.nn.softmax(y_att_1), axis=1))
    acc_final = acc_func(y_final)

var = tf.global_variables()
saver = tf.train.Saver(var) 

max_acc = 0.0

exist_frm = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint('subnet/'))
    
    for epoch in range(1):
        # training
        batch_x = np.zeros(shape=(BATCH_SIZE, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), dtype=np.float32)
        batch_y = np.zeros(shape=BATCH_SIZE, dtype=np.float32)

        bidx = 0

        total_pred = []

        bbox = []
        paths = []

        BBOX = {}
        video_name = ''

        n_train = len(val_files)
        for idx, val_file in enumerate(val_files):
            voxel, cls = read_test(val_file, frame_dir)
            paths.append(' '.join(val_file.split(' ')[:2]))
            
            batch_x[bidx] = voxel
            batch_y[bidx] = cls
            bidx += 1

            if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == n_train:
                #X.append(copy.deepcopy(batch_x[:bidx]))
                feeds = {x: batch_x[:bidx], y: batch_y[:bidx], training: False}
                ac_1, score = sess.run([AC_1, y_final], feed_dict=feeds)

                for b in range(bidx):
                    vname = paths[b].split('/')[0]
                    if len(video_name)>0 and video_name != vname:

                        bbox, BBOX = smoothing(bbox, BBOX, threshold)
                        
                    video_name = vname

                    heatmaps = np.zeros([FRAMES, *ac_1[b, 0].shape])
                    heatmaps[0] = ac_1[b, 0]
                    heatmaps[-1] = ac_1[b, 1]

                    for i in range(FRAMES):
                        heatmaps[i] = 1/15 * ( (15-i)*heatmaps[0] + i*heatmaps[-1] )

                        frame_name = str("%s %s" % (paths[b].split(' ')[0], str(int(paths[b].split(' ')[1])+i+1).zfill(4)))
                        if frame_name not in BBOX.keys():
                            BBOX[frame_name] = [np.zeros(ac_1[b, 0].shape), np.zeros(num_class), 0] # AC, score, counter
                        BBOX[frame_name][0] += heatmaps[i]
                        BBOX[frame_name][1] += score[b]
                        BBOX[frame_name][2] += 1

                    if (idx+1) == n_train:
                        bbox, BBOX = smoothing(bbox, BBOX, threshold)


                # reset batch
                bidx = 0
                paths.clear()
                #break

        with open('bbox_prediction.txt', 'w') as f:
            f.write('\n'.join(bbox))

