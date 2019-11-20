from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import cv2

def make_shuffle_idx(n):
    random.seed()
    order = list(range(n))
    random.shuffle(order)
    return order

def deepnn(x, x_2):

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_conv1_2 = tf.nn.relu(conv2d(x_2, W_conv1) + b_conv1)
    with tf.name_scope('conv1_'):
        W_conv1_ = weight_variable([5, 5, 32, 32])
        b_conv1_ = bias_variable([32])
        h_conv1_ = tf.nn.relu(conv2d(h_conv1, W_conv1_) + b_conv1_)
        h_conv1_2_ = tf.nn.relu(conv2d(h_conv1_2, W_conv1_) + b_conv1_)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
      h_pool1 = max_pool_2x2(h_conv1_)
      h_pool1_2 = max_pool_2x2(h_conv1_2_)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_conv2_2 = tf.nn.relu(conv2d(h_pool1_2, W_conv2) + b_conv2)
    with tf.name_scope('conv2_'):
        W_conv2_ = weight_variable([5, 5, 64, 64])
        b_conv2_ = bias_variable([64])
        h_conv2_ = tf.nn.relu(conv2d(h_conv2, W_conv2_) + b_conv2_)
        h_conv2_2_ = tf.nn.relu(conv2d(h_conv2_2, W_conv2_) + b_conv2_)

    # Second pooling layer.
    with tf.name_scope('pool2'):
      h_pool2 = max_pool_2x2(h_conv2_)
      h_pool2_2 = max_pool_2x2(h_conv2_2_)

    # Third convolutional layer -- maps 64 feature maps to 128.
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([5, 5, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_conv3_2 = tf.nn.relu(conv2d(h_pool2_2, W_conv3) + b_conv3)
    with tf.name_scope('conv3_'):
        W_conv3_ = weight_variable([5, 5, 128, 128])
        b_conv3_ = bias_variable([128])
        h_conv3_ = tf.nn.relu(conv2d(h_conv3, W_conv3_) + b_conv3_)
        h_conv3_2_ = tf.nn.relu(conv2d(h_conv3_2, W_conv3_) + b_conv3_)

    # Third pooling layer.
    with tf.name_scope('pool3'):
      h_pool3 = max_pool_2x2(h_conv3_)
      h_pool3_2 = max_pool_2x2(h_conv3_2_)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
      W_fc1 = weight_variable([4 * 4 * 128, 500])
      b_fc1 = bias_variable([500])

      h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 128])
      h_pool3_flat_2 = tf.reshape(h_pool3_2, [-1, 4 * 4 * 128])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
      h_fc1_2 = tf.nn.relu(tf.matmul(h_pool3_flat_2, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
      keep_prob = tf.placeholder(tf.float32, name='keep_prob')
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
      W_fc2 = weight_variable([500, 10])
      b_fc2 = bias_variable([10])

      y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob, h_fc1, h_fc1_2


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main():
    # Create the model
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
    x_2 = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x_2')


    # Define loss and optimizer
    y_ = tf.placeholder(tf.int64, [None])

    # Build the graph for the deep net
    y_conv, keep_prob, h_fc1, h_fc1_2 = deepnn(x, x_2)

    y = tf.nn.softmax(y_conv, name='softmax')
    tf.add_to_collection('pred_network', y)

    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
        mse_loss = tf.losses.mean_squared_error(h_fc1, h_fc1_2)
    cross_entropy = tf.reduce_mean(cross_entropy)
    mse_loss = tf.reduce_mean(mse_loss)

    loss = tf.add(cross_entropy, mse_loss)

    with tf.name_scope('adam_optimizer'):
      train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    saver = tf.train.Saver()

    batch_size = 64
    image = np.zeros((batch_size, 28, 28, 1), np.float32)
    image_2 = np.zeros((batch_size, 28, 28, 1), np.float32)
    label = np.zeros((batch_size), np.int32)

    f_train = open('mnist_train_pair.txt', 'r')
    f_test = open('mnist_test_pair.txt', 'r')
    lines = f_train.readlines()
    training_shuffle_idx = make_shuffle_idx(len(lines))
    lines = [lines[i] for i in training_shuffle_idx]
    image_num = len(lines)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        k = 0
        for i in range(10000):
            for j in range(batch_size):
                if k == image_num:
                    k = 0
                image_path, image_path_2, image_label = lines[k].strip('\n').split(' ')
                image[j, :, :, 0] = cv2.imread(image_path, 0).astype(np.float32)
                image_2[j, :, :, 0] = cv2.imread(image_path_2, 0).astype(np.float32)
                label[j] = int(image_label)
                k = k+1
            image = (image - 127.5)/128
            image_2 = (image_2 - 127.5)/128
            _, loss1, loss2 = sess.run([train_step, cross_entropy, mse_loss], feed_dict={x: image, x_2: image_2, y_: label, keep_prob: 0.5})
            if i % 10 == 0:
                test_accuracy = accuracy.eval(feed_dict={x: image, y_: label, keep_prob: 1.0})
                print('batch accuracy=', test_accuracy, '\t', 'cross_entropy=', loss1, '\t', 'mse_loss=', loss2)
        saver.save(sess, "./save/conv_model")

        test_accuracy=0
        test_image = np.zeros((1, 28, 28, 1), np.float32)

        for i in range(10000):
            image_path, image_path_2, image_label = f_test.readline().strip('\n').split(' ')
            test_image[0, :, :, 0] = cv2.imread(image_path, 0).astype(np.float32)
            test_accuracy+=accuracy.eval(feed_dict={x: test_image, y_: int(image_label), keep_prob: 1.0})
        print('test accuracy %g' % (test_accuracy*0.0001))
if __name__ == '__main__':
    main()
