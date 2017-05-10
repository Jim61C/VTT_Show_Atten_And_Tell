# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import tensorflow as tf


class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_feature=[1, 2048], dim_embed=512, dim_hidden=1024, n_time_step=16, 
                  prev2out=True, ctx2out=False, alpha_c=0.0, selector=False, dropout=True):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM. 
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """
        
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])
    
    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x, w

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)  
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            alpha = tf.nn.softmax(out_att)  
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            return context, alpha
  
    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
            context = tf.multiply(beta, context, name='selected_context') 
            return context, beta
  
    def _decode_lstm(self, x, h, dropout=False, reuse=False): #remove context
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h
            """
            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)
            """
            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits
        
    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x, 
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)


    def build_vtt_model(self, max_len=20, mode='train'):
        features = self.features
        captions = self.captions
        mask = tf.to_float(tf.not_equal(captions[:, 1:], self._null))

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')
        
        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)

        # LSTM loss
        # input features are ground truth text embeddings [0..T], captions are [1..T+1]
        lstm_loss = 0.0
        '''
        if mode == 'train':
            captions_in = captions[:, :self.T]      
            captions_out = captions[:, 1:]  
            mask = tf.to_float(tf.not_equal(captions_out, self._null))
            x_gt, _ = self._word_embedding(inputs=captions_in)
            for t in range(self.T):
                with tf.variable_scope('lstm', reuse=(t!=0)):
                    _, (c, h) = lstm_cell(inputs=x_gt[:,t,:], state=[c, h])

                logits_gt = self._decode_lstm(x_gt[:,t,:], h, dropout=self.dropout, reuse=(t!=0))
                lstm_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_gt, labels=captions_out[:, t]) * mask[:, t])
            lstm_loss /= tf.to_float(tf.shape(features)[0])
        '''

        # GAN loss
        # input is the sentence that LSTM generated (in test mode)
        xs = []
        for t in range(max_len):
            if t == 0:
                x, w = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
                xs.append(x)
            else:
                x, _ = self._word_embedding(inputs=sampled_word, reuse=True)
                xs.append(tf.matmul(probs, w))

            """
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha)
            
            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0)) 
                beta_list.append(beta)
            """
            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=x, state=[c, h])

            logits = self._decode_lstm(x, h, reuse=(t!=0))
            probs = tf.nn.softmax(logits)
            sampled_word = tf.argmax(logits, 1)       
            sampled_word_list.append(sampled_word)  

            if t<max_len-1:
                #logits_gt = self._decode_lstm(x_gt[:,t,:], h, dropout=self.dropout, reuse=(t!=0))
                lstm_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=captions[:, t+1]) * mask[:, t])
        
        lstm_loss /= tf.to_float(tf.shape(features)[0])

        xs = tf.reshape(tf.transpose(xs, perm = (1, 0, 2)), shape = (-1, max_len * self.M))

        alphas = [] #tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
        betas = [] #tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        
        # input of GAN
        image_feature = tf.squeeze(features)                             # (128, 2048)
        generated_input = tf.concat([xs, image_feature], 1, name = "generated_input")

        input_cap, _ = self._word_embedding(inputs=captions, reuse=True)  
        input_cap = tf.reshape(input_cap, shape = (-1, max_len * self.M))             # (128, 8704)
        groundtruth_input = tf.concat([input_cap, image_feature], 1, name = "gt_input")   # (128, 8704+2048)

        # GAN loss
        X_dim = groundtruth_input.get_shape().as_list()[1] #[17*512], #words * #embed, TODO::  + 2048  #mnist.train.images.shape[1]
        h_dim = 128

        with tf.variable_scope("Discriminator_network"):
            
            """ Discriminator Net model """
            D_W1 = tf.get_variable('D_W1', [max_len * self.M + self.D, h_dim], initializer=self.weight_initializer) #tf.Variable(self.xavier_init([X_dim, h_dim]))
            D_b1 = tf.Variable(tf.zeros(shape=[h_dim,]))

            D_W2 = tf.get_variable('D_W2', [h_dim, 1], initializer=self.weight_initializer) 
            D_b2 = tf.Variable(tf.zeros(shape=[1,]))

            theta_D = [D_W1, D_W2, D_b1, D_b2]

            """ Discriminator """
            #print groundtruth_input.shape
            #print D_W1.shape
            D_h1 = tf.nn.relu(tf.matmul(groundtruth_input, D_W1) + D_b1)
            D_logit_real = tf.matmul(D_h1, D_W2) + D_b2
            D_real = tf.nn.sigmoid(D_logit_real)

            D_h1 = tf.nn.relu(tf.matmul(generated_input, D_W1) + D_b1)
            D_logit_fake = tf.matmul(D_h1, D_W2) + D_b2
            D_fake = tf.nn.sigmoid(D_logit_fake)

        """ Discriminator loss """
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake

        """ Generator loss """
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        return alphas, betas, sampled_captions, D_loss, G_loss, lstm_loss 
