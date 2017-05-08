# Reference 1 (main):
#   Conditional GAN: https://github.com/wiseodd/generative-models/blob/master/GAN/conditional_gan/cgan_tensorflow.py
# Reference 2:
#   LSTM-GAN: https://github.com/vangaa/lstm_gan/blob/master/train.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def discriminator(inputs):
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def gan(input_cap, generated_cap, features_batch):
    global D_W1, D_W2, D_b1, D_b2, D_h1
    image_feature = features_batch
    image_feature = tf.squeeze(image_feature)                       # (128, 2048)
    #print "image_feature.shape ", image_feature.shape


    # concatenate image_feature and text vector
    gt_shape = input_cap.get_shape().as_list()
    gen_shape = generated_cap.get_shape().as_list()

    input_cap = tf.reshape(input_cap, [gt_shape[0], -1])             # (128, 8704)
    generated_cap = tf.reshape(generated_cap, [gen_shape[0], -1])    # (128, 8704)

    #print "input_cap.shape ", input_cap.shape
    #print "generated_cap.shape ", generated_cap.shape

    groundtruth_input = tf.concat([input_cap, image_feature], 1)
    generated_input = tf.concat([generated_cap, image_feature], 1)

    # variables
    X_dim = groundtruth_input.get_shape().as_list()[1] #[17*512], #words * #embed, TODO::  + 2048  #mnist.train.images.shape[1]
    h_dim = 128
    #print "X_dim = ", X_dim


    """ Discriminator Net model """
    X = groundtruth_input #tf.placeholder(tf.float32, shape=[None, 2048])

    D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_D = [D_W1, D_W2, D_b1, D_b2]


    """ D,G loss """
    G_sample = generated_input
    D_real, D_logit_real = discriminator(X)
    D_fake, D_logit_fake = discriminator(G_sample)

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_D)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    lossg = 0
    lossd = 0
    for it in range(1000):
        X_mb = groundtruth_input.eval()
        Z_sample = generated_input.eval()
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: Z_sample})

        lossd = D_loss_curr
        lossg = G_loss_curr

        if it % 100 == 0:
            print('GAN Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G loss: {:.4}'.format(G_loss_curr))
            print ""
    print ""
    return lossg, lossd
