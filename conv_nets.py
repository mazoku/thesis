from __future__ import division

import numpy as np

import progressbar
import tensorflow as tf


def load_dataset():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    return mnist


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def simple_tf():
    mnist = load_dataset()

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    max_it = 1000
    widgets = ["Training: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=max_it, widgets=widgets)
    pbar.start()
    for i in range(max_it):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        pbar.update(i)
    pbar.finish()

    # printing precision
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


def advanced_tf():
    mnist = load_dataset()

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # first convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # training and evaluating
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # init = tf.initialize_all_variables()
    # sess = tf.Session()
    # sess.run(init)
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


def read_and_decode_single_example(filename, data_size, labels_size):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep
    # size down
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([labels_size], tf.int64),
            'image': tf.FixedLenFeature([data_size], tf.int64)
            # 'label': tf.VarLenFeature(tf.int64),
            # 'image': tf.VarLenFeature(tf.int64)
        })
    # now return the converted data
    label = features['label']
    image = features['image']

    # resizing if rewuired
    # if resize_labels_fac != 0:
    #     pass
    return label, image


def train_batches(tfrecords_path, data_shape, labels_shape):
    data_size = np.prod(data_shape)
    labels_size = np.prod(labels_shape)
    # get single examples
    label, image = read_and_decode_single_example(tfrecords_path, data_size, labels_size)
    image = tf.cast(image, tf.float32) / 255.
    # groups examples into batches randomly
    images_batch, labels_batch = tf.train.shuffle_batch([image, label], batch_size=128, capacity=2000,
                                                        min_after_dequeue=1000)

    # simple model
    w = tf.get_variable('w1', [data_size, labels_size])
    y_pred = tf.matmul(images_batch, w)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, labels_batch)

    # for monitoring
    loss_mean = tf.reduce_mean(loss)

    train_op = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    it = 1
    while True:
        print 'iteration # %i' % it
        # pass it in through the feed_dict
        _, loss_val = sess.run([train_op, loss_mean])
        print loss_val
        it += 1

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    advanced_tf()

    tfrecords_path = '/home/tomas/Data/medical/dataset/gt/slicewise/data.tfrecords'
    data_size = (60, 60)
    labels_shape = (20, 20)
    train_batches(tfrecords_path, data_size, labels_shape)