import os
from sklearn.utils import shuffle
from time import ctime
import tensorflow as tf
import scipy.io as sio
import numpy as np
from scipy.misc import imresize
import os
import copy
from model import *
from utils import *

from argparse import ArgumentParser

parser = ArgumentParser(description='Process some integers.')
parser.add_argument('-m', '--mode', type=str, default='train_main',
                    help='training mode')
# the dictionay containing training/test list files 
parser.add_argument('-t', '--trainingfile', type=str,
                    help='data list')
parser.add_argument('-p', '--path', type=str,
                    help='data path')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# UCF101 frame shape (240, 320, 3)#

HEIGHT = 128
WIDTH = 171
FRAMES = 16
RE_SIZE = 240 
CROP_SIZE = 112 
CHANNELS = 3
BATCH_SIZE = 16 

num_class=24

trainingfile = os.path.join(args.trainingfile, 'train.txt')
print(trainingfile)

# Demo of training on UCF101
with open(trainingfile, 'r') as f:
    lines = f.readlines()
tr_files = [line for line in lines if len(line) > 0]

np.random.shuffle(tr_files)
num = len(tr_files)
tr_files, val_files = tr_files[: int(3*num/4)], tr_files[int(3*num/4):]

# Define placeholders
x = tf.placeholder(tf.float32, shape=(None, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), name='input_x')
y = tf.placeholder(tf.int64, None, name='input_y')
training = tf.placeholder(tf.bool, name='training')

# Define the C3D model for UCF101
inputs = x - tf.constant([96.6], dtype=tf.float32, shape=[1, 1, 1, 1, 1])
#inputs = tf.scalar_mul(2/255.0, x) - tf.constant([1.0], dtype=tf.float32, shape=[1, 1, 1, 1, 1])
logits, [y_att_1, _, _] = c3d(inputs=inputs, num_class=num_class, training=training)
labels = tf.one_hot(y, num_class, name='labels')

def acc_func(logits, labels=labels, y=y):
    if len(logits.get_shape().as_list()) > 2:
        logits = tf.reduce_mean(tf.nn.softmax(y_att_1), axis=1)
        #logits = tf.reduce_mean(tf.nn.relu(logits), axis=1)
        #logits = tf.reduce_mean(logits, axis=1)
        #_, T, c = logits.get_shape().as_list()
        #y = tf.reshape(y, [-1, 1])
        #y = tf.tile(y, [1, T]) 

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

# Define loss function
def loss_func(logits, labels=labels, name='loss'):
    if len(logits.get_shape().as_list()) > 2:
        #logits = tf.reduce_sum(tf.nn.relu(logits), axis=1)
        #logits = tf.reduce_sum(logits, axis=1)
        _, T, c = logits.get_shape().as_list()
        labels = tf.reshape(labels, [-1, 1, c])
        labels = tf.tile(labels, [1, T, 1]) 

        labels = tf.reshape(labels, [-1, c])
        logits = tf.reshape(logits, [-1, c])

    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name=name)
    #return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name=name)

loss = loss_func(logits, name='loss')
loss_subnet_att_1 = loss_func(y_att_1, name='loss_subnet_att_1')

# Define optimizer
def opt_func(loss, var, lr=0.003):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=lr, global_step=global_step, decay_steps=300, 
                                        decay_rate=0.98, staircase=True)
    return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, var_list=var)


train_opt = opt_func(loss, tf.global_variables(), lr=0.001)

# Optimize subnet1 all
var_list = tf.global_variables()
var_list = [v for v in var_list if 'subnet_1' in v.name]
train_opt_subnet_1 = opt_func(loss_subnet_att_1, var_list, lr=0.003)

train_opt_all = opt_func(loss+loss_subnet_att_1, tf.global_variables(), lr=0.001)

if args.mode == 'train_main':
    var = tf.global_variables()
    var = [v for v in var if 'c3d_output' not in v.name and 'c3d' in v.name]

    # save & restore all variable
    saver = tf.train.Saver()

    # restore variables except 'c3d_output'
    saver1 = tf.train.Saver(var)

    max_acc = 0.0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver1.restore(sess, tf.train.latest_checkpoint('pretrain/'))

        n_train = len(tr_files)
        #for epoch in range(100):

        min_lss = 10.0
        
        for epoch in range(30):
            # training
            tr_files = shuffle(tr_files)
            batch_x = np.zeros(shape=(BATCH_SIZE, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), dtype=np.float32)
            batch_y = np.zeros(shape=BATCH_SIZE, dtype=np.float32)

            bidx = 0
            total_lss = []
            total_acc = []
            n_train = len(tr_files)
            for idx, tr_file in enumerate(tr_files):
                voxel, cls = read_train(tr_file)
                
                batch_x[bidx] = voxel
                batch_y[bidx] = cls
                bidx += 1

                if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == n_train:
                    feeds = {x: batch_x[:bidx], y: batch_y[:bidx], training: True}
                    _, lss, acc = sess.run([train_opt, loss, acc_opt], feed_dict=feeds)

                    print('training %04d %04d/%04d/%04d, loss: %.3f, acc: %.2f' % (epoch, idx / BATCH_SIZE, idx, n_train, lss, acc))
                    total_lss.append(lss)
                    total_acc.append(acc)

                    # reset batch
                    bidx = 0
                
            
            print('training lss = %.3f' % np.mean(np.array(total_lss)))
            print('training acc = %.2f' % np.mean(np.array(total_acc)))

            # validation
            val_files = shuffle(val_files)

            bidx = 0

            total_lss = []
            total_acc = []
            n_train = len(val_files)
            for idx, val_file in enumerate(val_files):
                voxel, cls = read_train(val_file)
                
                batch_x[bidx] = voxel
                batch_y[bidx] = cls
                bidx += 1

                if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == n_train:
                    feeds = {x: batch_x[:bidx], y: batch_y[:bidx], training: False}
                    lss, acc = sess.run([loss, acc_opt], feed_dict=feeds)

                    total_lss.append(lss)
                    total_acc.append(acc)

                    # reset batch
                    bidx = 0

            print('validation lss = %.3f' % np.mean(np.array(total_lss)))
            print('validation acc = %.2f' % np.mean(np.array(total_acc)))
            
            if np.mean(np.array(total_lss)) <= min_lss:
                saver.save(sess, 'model/ucf24_model', global_step=epoch)
                min_lss = np.mean(np.array(total_lss))

elif args.mode == 'train_subnet':
    var = tf.global_variables()
    var = [v for v in var if 'c3d' in v.name]

    # save & restore all variable
    saver = tf.train.Saver()

    # restore partial variables
    saver1 = tf.train.Saver(var)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver1.restore(sess, tf.train.latest_checkpoint('model/'))

        min_lss = 10.0
        for epoch in range(30):
            # training
            tr_files = shuffle(tr_files)
            batch_x = np.zeros(shape=(BATCH_SIZE, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), dtype=np.float32)
            batch_y = np.zeros(shape=BATCH_SIZE, dtype=np.float32)

            bidx = 0
            total_lss_att = []
            total_acc_att = []
            total_acc = []
            total_acc_final = []
            n_train = len(tr_files)

            for idx, tr_file in enumerate(tr_files):
                voxel, cls = read_train(tr_file)
                
                batch_x[bidx] = voxel
                batch_y[bidx] = cls
                bidx += 1

                if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == n_train:
                    feeds = {x: batch_x[:bidx], y: batch_y[:bidx], training: True}
                    _, lss_att, acc_att, acc, acc_f = sess.run([train_opt_subnet_1, loss_subnet_att_1, acc_subnet_att_1,
                                                        acc_opt, acc_final], feed_dict=feeds)

                    print('training %04d %04d/%04d/%04d, loss_att: %.3f, acc_att: %.2f, acc: %.2f, acc_final: %.2f' % (epoch, idx / BATCH_SIZE, idx, n_train, lss_att, acc_att, acc, acc_f))
                    total_lss_att.append(lss_att)
                    total_acc_att.append(acc_att)
                    total_acc.append(acc)
                    total_acc_final.append(acc_f)

                    # reset batch
                    bidx = 0
                
            
            print('training lss att = %.3f' % np.mean(np.array(total_lss_att)))
            print('training acc att = %.2f' % np.mean(np.array(total_acc_att)))
            print('training acc = %.2f' % np.mean(np.array(total_acc)))
            print('training acc final = %.2f' % np.mean(np.array(total_acc_final)))

            # validation
            val_files = shuffle(val_files)

            bidx = 0

            total_lss_att = []
            total_acc_att = []
            total_acc = []
            total_acc_final = []

            n_train = len(val_files)
            for idx, val_file in enumerate(val_files):
                voxel, cls = read_train(val_file)
                
                batch_x[bidx] = voxel
                batch_y[bidx] = cls
                bidx += 1

                if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == n_train:
                    feeds = {x: batch_x[:bidx], y: batch_y[:bidx], training: False}
                    lss_att, acc_att, acc, acc_f = sess.run([loss_subnet_att_1, acc_subnet_att_1, acc_opt,
                        acc_final], feed_dict=feeds)

                    total_lss_att.append(lss_att)
                    total_acc_att.append(acc_att)
                    total_acc.append(acc)
                    total_acc_final.append(acc_f)

                    # reset batch
                    bidx = 0

            print('validation lss att = %.3f' % np.mean(np.array(total_lss_att)))
            print('validation acc att = %.2f' % np.mean(np.array(total_acc_att)))
            print('validation acc = %.2f' % np.mean(np.array(total_acc)))
            print('validation acc final = %.2f' % np.mean(np.array(total_acc_final)))

            new_min_lss = np.mean(np.array(total_lss_att))
            if new_min_lss <= min_lss:
                min_lss = new_min_lss
                saver.save(sess, 'subnet/ucf24_model', global_step=epoch)

