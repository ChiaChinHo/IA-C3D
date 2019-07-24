import tensorflow as tf

def c3d(inputs, num_class, training):
    """
    C3D network for video classification
    :param inputs: Tensor inputs (batch, depth=16, height=112, width=112, channels=3), should be means subtracted
    :param num_class: A scalar for number of class
    :param training: A boolean tensor for training mode (True) or testing mode (False)
    :return: Output tensor
    """

    net = tf.layers.conv3d(inputs, filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu, name='c3d_conv1')
    net = tf.layers.max_pooling3d(net, pool_size=(1, 2, 2), strides=(1, 2, 2), padding='SAME')

    net = tf.layers.conv3d(net, filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu, name='c3d_conv2')
    net = tf.layers.max_pooling3d(net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.conv3d(net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu, name='c3d_conv3a')
    net = tf.layers.conv3d(net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu, name='c3d_conv3b')
    net = tf.layers.max_pooling3d(net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.conv3d(net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu, name='c3d_conv4a')
    net = tf.layers.conv3d(net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu, name='c3d_conv4b')
    net = tf.layers.max_pooling3d(net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.conv3d(net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu, name='c3d_conv5a')
    net = tf.layers.conv3d(net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu, name='c3d_conv5b')

    with tf.variable_scope("subnet_1"):
        y_att_1, AC_1, ATT = subnet_w_inception(net, num_class, 512, training)

    net = tf.layers.max_pooling3d(net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, units=4096, activation=tf.nn.relu, name='c3d_fc1')
    net = tf.layers.dropout(net, rate=0.5, training=training)

    net = tf.layers.dense(net, units=4096, activation=tf.nn.relu, name='c3d_fc2')
    net = tf.layers.dropout(net, rate=0.5, training=training)

    net = tf.layers.dense(net, units=num_class, activation=None, name='c3d_output')

    return net, [y_att_1, AC_1, ATT]

def spatial_normalize(att):
    T, h, w, c = att.get_shape().as_list()[1:]
    att = tf.reshape(att, [-1, T, h*w, c])
    
    # spatial softmax
    att = tf.nn.softmax(att, dim=2)
    att = tf.reshape(att, [-1, T, h, w, c])

    return att

def non_linear(att, conf, num_class):
    T, h, w = att.get_shape().as_list()[1:4]

    #AC = tf.multiply(att, tf.nn.tanh(conf))
    #y_att = tf.reduce_sum(tf.reshape(tf.multiply(att, tf.nn.tanh(conf)), [-1, T, h*w, num_class]), axis=2)

    AC = tf.multiply(att, tf.nn.sigmoid(conf))
    y_att = tf.reduce_sum(tf.reshape(tf.multiply(att, tf.nn.sigmoid(conf)), [-1, T, h*w, num_class]), axis=2)

    return y_att, AC
    
def subnet_w_inception(inputs, num_class, filters, training):
    
    att = tf.layers.dropout(inputs, rate=0.5, training=training, name='att_dropout1')
    att_conv1 = tf.layers.conv3d(att, filters=filters, kernel_size=1, padding='SAME', name='att_conv1')
    att_conv1 = tf.layers.dropout(att_conv1, rate=0.5, training=training, name='att_s1_dropout')

    att_conv2 = tf.layers.conv3d(att, filters=filters, kernel_size=1, activation=tf.nn.relu, padding='SAME', name='att_conv2a')
    att_conv2 = tf.layers.dropout(att_conv2, rate=0.5, training=training, name='att_s2_dropout')
    att_conv2 = tf.layers.conv3d(att_conv2, filters=filters, kernel_size=(1, 3, 3), padding='SAME', name='att_conv2b')
    
    att_conv3 = tf.layers.conv3d(att, filters=filters, kernel_size=1, activation=tf.nn.relu, padding='SAME', name='att_conv3a')
    att_conv3 = tf.layers.dropout(att_conv3, rate=0.5, training=training, name='att_s3_dropout')
    att_conv3 = tf.layers.conv3d(att_conv3, filters=filters, kernel_size=(1, 3, 3), padding='SAME', name='att_conv3b')

    att = tf.concat([att_conv1, att_conv2, att_conv3], -1)
    att = tf.layers.batch_normalization(att, training=training)
    att = tf.nn.relu(att)
    att = tf.layers.conv3d(att, filters=num_class, kernel_size=1, padding='SAME', name='att_conv4')

    att = spatial_normalize(att)
    T, h, w, c = att.get_shape().as_list()[1:]

    conf = tf.layers.conv3d(inputs, filters=filters, kernel_size=1, padding='SAME', name='conf_conv1')
    conf = tf.layers.batch_normalization(conf, training=training)
    conf = tf.nn.relu(conf)
    conf = tf.layers.conv3d(conf, filters=num_class, kernel_size=(1, 3, 3), padding='SAME', name='conf_conv2')

    y_att, AC = non_linear(att, conf, num_class)

    return y_att, AC, att
