import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os
import cPickle as pickle
from scipy import ndimage
from utils import *
from bleu import evaluate

class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17)
                - image_idxs: Indices for mapping caption to image of shape (400000, )
                - word_to_idx: Mapping dictionary from word to index
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path
            - model_path: String; model path for saving
            - test_model: String; model path for test
        """

        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        self.reset_embedding = kwargs.pop('reset_embedding', False)
        self.use_gan = kwargs.pop('use_gan', False)

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)


    def train(self):
        # train/val dataset
        n_examples = self.data['captions'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
        features = self.data['features']
        captions = self.data['captions']
        image_idxs = self.data['image_idxs']
        val_features = self.val_data['features']
        n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))

        # build graphs for training model and sampling captions
        loss = self.model.build_model()
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            tf.get_variable_scope().reuse_variables()
            _, _, generated_captions = self.model.build_sampler(max_len=17)

        # train op
        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        # summary op
        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        for grad, var in grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name+'/gradient', grad)

        summary_op = tf.summary.merge_all()

        print "The number of epoch: %d" %self.n_epochs
        print "Data size: %d" %n_examples
        print "Batch size: %d" %self.batch_size
        print "Iterations per epoch: %d" %n_iters_per_epoch

        config = tf.ConfigProto(allow_soft_placement = True)
        #config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print "Start training with pretrained Model.."
                saver.restore(sess, self.pretrained_model)

            if self.reset_embedding == True:
                var_list = []
                var_list += tf.contrib.framework.get_variables_by_name('word_embedding/w')
                var_list += tf.contrib.framework.get_variables_by_name('logits/w_out')
                var_list += tf.contrib.framework.get_variables_by_name('logits/b_out')
                print "variable before re-initialization"
                for var in var_list:
                    print var.eval()
                sess.run(tf.variables_initializer(var_list))
                print "variable after re-initialization"
                for var in var_list:
                    print var.eval()

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            for e in range(self.n_epochs):
                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                image_idxs = image_idxs[rand_idxs]

                for i in range(n_iters_per_epoch):
                    print i
                    captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
                    image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
                    features_batch = features[image_idxs_batch]
                    feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l

                    # write summary for tensorboard visualization
                    if i % 10 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, e*n_iters_per_epoch + i)

                    '''
                    # adversarial training
                    if self.use_gan:
                        # generated captions
                        gen_caps = sess.run(generated_captions, feed_dict)                          #(128, 17)
                        generated_cap = self.model._word_embedding(inputs=gen_caps, reuse=True)     #(128, 17, 512)

                        # ground truth captions
                        ground_truths = captions_batch #captions[image_idxs == image_idxs_batch[0]] #(128, 17)
                        input_cap = self.model._word_embedding(inputs=ground_truths, reuse=True)    #(128, 17, 512)

                        lossg, lossd = gan(input_cap, generated_cap, features_batch)  # features_batch (n, 1, 2048)
                        curr_loss_generator += lossg
                        curr_loss_discriminator += lossd
                    '''

                    if (i+1) % self.print_every == 0:
                        print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l)
                        ground_truths = captions[image_idxs == image_idxs_batch[0]]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print "Ground truth %d: %s" % (j+1, gt)
                        gen_caps = sess.run(generated_captions, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print "Generated caption: %s\n" %decoded[0]

                print "Previous epoch loss: ", prev_loss
                print "Previous epoch generator loss: ", prev_lossg
                print "Previous epoch discriminator loss: ", prev_lossd

                print "Current epoch loss: ", curr_loss
                print "Current epoch generator loss: ", curr_loss_generator
                print "Current epoch discriminator loss: ", curr_loss_discriminator
                print "Elapsed time: ", time.time() - start_t

                prev_loss = curr_loss
                curr_loss = 0

                # print out BLEU scores and file write
                if self.print_bleu:
                    all_gen_cap = np.ndarray((val_features.shape[0], 20))
                    for i in range(n_iters_val):
                        features_batch = val_features[i*self.batch_size:(i+1)*self.batch_size]
                        feed_dict = {self.model.features: features_batch}
                        gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
                        all_gen_cap[i*self.batch_size:(i+1)*self.batch_size] = gen_cap

                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, "./data_MSRVTT/val/val.candidate.captions.pkl")
                    #######
                    scores = evaluate(data_path='./data_MSRVTT', split='val', get_scores=True)
                    #######
                    write_bleu(scores=scores, path=self.model_path, epoch=e)

                # save model's parameters
                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                    print "model-%s saved." %(e+1)


    def test(self, data, split='train', attention_visualization=False, save_sampled_captions=True, save_path='./data'):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17)
            - image_idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        features = data['features']

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=17)    # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
            feed_dict = { self.model.features: features_batch }
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)

            if attention_visualization:
                for n in range(10):
                    print "Sampled Caption: %s" %decoded[n]

                    # Plot original image
                    img = ndimage.imread(image_files[n])
                    plt.clf()
                    plt.subplot(4, 5, 1)
                    plt.imshow(img)
                    plt.axis('off')

                    # Plot images with attention weights
                    words = decoded[n].split(" ")
                    for t in range(len(words)):
                        if t > 18:
                            break
                        plt.subplot(4, 5, t+2)
                        plt.text(0, 1, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
                        plt.imshow(img)
                        alp_curr = alps[n,t,:].reshape(14,14)
                        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                        plt.imshow(alp_img, alpha=0.85)
                        plt.axis('off')
                    plt.show()
                    plt.savefig('./figs/model_' + str(n))

            if save_sampled_captions:
                all_sam_cap = np.ndarray((features.shape[0], 20))
                num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
                for i in range(num_iter):
                    features_batch = features[i*self.batch_size:(i+1)*self.batch_size]
                    feed_dict = { self.model.features: features_batch }
                    all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)
                all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
                save_pickle(all_decoded, save_path+"/%s/%s.candidate.captions.pkl" %(split,split))


    def optimistic_restore(self, session, save_file):
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)

    def train_gan(self):
        # train/val dataset
        n_examples = self.data['captions'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
        features = self.data['features']
        captions = self.data['captions']
        image_idxs = self.data['image_idxs']
        val_features = self.val_data['features']
        n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))

        # build graphs for training model and sampling captions
        _, _, generated_captions, D_loss, G_loss = self.model.build_vtt_model(max_len=17)
        #with tf.variable_scope(tf.get_variable_scope()) as scope:
        #    tf.get_variable_scope().reuse_variables()
        #    _, _, generated_captions = self.model.build_sampler(max_len=17)

        # train op
        with tf.name_scope('Discriminator_optimizer'):
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator_network')
            D_optimizer = self.optimizer(learning_rate=self.learning_rate)
            D_grads_and_vars = D_optimizer.compute_gradients(D_loss, var_list)
            D_train_op = D_optimizer.apply_gradients(grads_and_vars=D_grads_and_vars)

        with tf.name_scope('Generator_optimizer'):
            G_optimizer = self.optimizer(learning_rate=self.learning_rate)
            G_grads_and_vars = G_optimizer.compute_gradients(G_loss, tf.trainable_variables())
            G_train_op = G_optimizer.apply_gradients(grads_and_vars=G_grads_and_vars)

        # summary op
        tf.summary.scalar('batch_generator_loss', G_loss)
        tf.summary.scalar('batch_discriminator_loss', D_loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        for grad, var in D_grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name+'/D_gradient', grad)
        for grad, var in G_grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name+'/G_gradient', grad)
        summary_op = tf.summary.merge_all()

        print "The number of epoch: %d" %self.n_epochs
        print "Data size: %d" %n_examples
        print "Batch size: %d" %self.batch_size
        print "Iterations per epoch: %d" %n_iters_per_epoch

        config = tf.ConfigProto(allow_soft_placement = True)
        #config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print "Start training with pretrained Model.."
                self.optimistic_restore(sess, self.pretrained_model)

            if self.reset_embedding == True:
                var_list = []
                var_list += tf.contrib.framework.get_variables_by_name('word_embedding/w')
                var_list += tf.contrib.framework.get_variables_by_name('logits/w_out')
                var_list += tf.contrib.framework.get_variables_by_name('logits/b_out')
                print "variable before re-initialization"
                for var in var_list:
                    print var.eval()
                sess.run(tf.variables_initializer(var_list))
                print "variable after re-initialization"
                for var in var_list:
                    print var.eval()

            prev_loss = -1
            curr_loss = 0

            prev_D_loss = -1
            curr_D_loss = 0

            prev_G_loss = -1
            curr_G_loss = 0

            start_t = time.time()

            for e in range(self.n_epochs):
                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                image_idxs = image_idxs[rand_idxs]

                for i in range(n_iters_per_epoch):
                    #print i
                    captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
                    image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
                    features_batch = features[image_idxs_batch]
                    feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}
                    #print "features_batch.shape", features_batch.shape
                    #print "captions_batch.shape", captions_batch.shape
                    _, D_l, _, G_l, gen_caps = sess.run([D_train_op, D_loss, G_train_op, G_loss, generated_captions], feed_dict)

                    curr_loss += D_l + G_l
                    curr_D_loss += D_l
                    curr_G_loss += G_l

                    # write summary for tensorboard visualization
                    if i % 10 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, e*n_iters_per_epoch + i)

                    if (i+1) % self.print_every == 0:
                        print "\nTrain loss at epoch %d & iteration %d (mini-batch G): %.5f (D): %.5f " %(e+1, i+1, G_l, D_l)
                        '''
                        ground_truths = captions[image_idxs == image_idxs_batch[0]]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print "Ground truth  %d: %s" % (j+1, gt)
                        '''
                        ground_truth = captions_batch[0] #captions[image_idxs == image_idxs_batch[0]]
                        decoded = decode_captions(ground_truth, self.model.idx_to_word)
                        print "Ground truth: %s" % decoded
                        #gen_caps = sess.run(generated_captions, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print "Generated caption: %s\n" % decoded[0]

                print "Previous epoch loss: ", prev_loss
                print "Previous epoch generator loss: ", prev_G_loss
                print "Previous epoch discriminator loss: ", prev_D_loss

                print "Current epoch loss: ", curr_loss
                print "Current epoch generator loss: ", curr_G_loss
                print "Current epoch discriminator loss: ", curr_D_loss
                print "Elapsed time: ", time.time() - start_t

                prev_loss = curr_loss
                curr_loss = 0

                prev_D_loss = curr_D_loss
                curr_D_loss = 0

                prev_G_loss = curr_G_loss
                curr_G_loss = 0

                # print out BLEU scores and file write
                if self.print_bleu:
                    all_gen_cap = np.ndarray((val_features.shape[0], 17))
                    for i in range(n_iters_val):
                        features_batch = val_features[i*self.batch_size:(i+1)*self.batch_size]
                        feed_dict = {self.model.features: features_batch}
                        gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
                        all_gen_cap[i*self.batch_size:(i+1)*self.batch_size] = gen_cap

                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, "./data_MSRVTT/val/val.candidate.captions.pkl")
                    #######
                    scores = evaluate(data_path='./data_MSRVTT', split='val', get_scores=True)
                    #######
                    write_bleu(scores=scores, path=self.model_path, epoch=e)

                # save model's parameters
                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                    print "model-%s saved." %(e+1)

