import skvideo.io
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os
import cPickle as pickle
from scipy import ndimage
from utils import *
from bleu import evaluate
import sys


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
                - word_to_idx: Mapping di1ctionary from word to index
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
        self.use_tag = kwargs.pop('use_tag', False)
        self.data_path = kwargs.pop('data_path', './data_MSRVTT')

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
        if self.use_tag:
            tags = self.data['tags']
        captions = self.data['captions']
        image_idxs = self.data['image_idxs']
        val_features = self.val_data['features']
        if self.use_tag:
            val_tags = self.val_data['tags']
        n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))

        # build graphs for training model and sampling captions
        loss = self.model.build_model()
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            tf.get_variable_scope().reuse_variables()
            _, _, generated_captions = self.model.build_sampler(max_len=20)

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

        # config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True)
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

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            for e in range(self.n_epochs):
                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                image_idxs = image_idxs[rand_idxs]

                for i in range(n_iters_per_epoch):
                    #print "iteration: ", i+1, "self.print_every: ", self.print_every
                    captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
                    image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
                    features_batch = features[image_idxs_batch]
                    if self.use_tag:
                        tags_batch = tags[image_idxs_batch]
                        feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch, self.model.tags: tags_batch}
                    else:
                        feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l

                    # write summary for tensorboard visualization
                    if i % 10 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, e*n_iters_per_epoch + i)

                    if (i+1) % self.print_every == 0:
                        print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l)
                        ground_truths = captions[image_idxs == image_idxs_batch[0]]
                        #print len(image_idxs)
                        #print len(image_idxs_batch)
                        #print len(captions[image_idxs == image_idxs_batch[0]])
                        #print "ground_truths: ", ground_truths
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print "Ground truth %d: %s" %(j+1, gt)
                        gen_caps = sess.run(generated_captions, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print "Generated caption: %s\n" %decoded[0]

                    sys.stdout.flush()

                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                print "Elapsed time: ", time.time() - start_t
                prev_loss = curr_loss
                curr_loss = 0

                # print out BLEU scores and file write
                if self.print_bleu:
                    all_gen_cap = np.ndarray((val_features.shape[0], 20))
                    for i in range(n_iters_val):
                        features_batch = val_features[i*self.batch_size:(i+1)*self.batch_size]
                        if self.use_tag:
                            tags_batch = val_tags[i*self.batch_size:(i+1)*self.batch_size]
                            feed_dict = {self.model.features: features_batch, self.model.tags: tags_batch}
                        else:
                            feed_dict = {self.model.features: features_batch}
                        gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
                        all_gen_cap[i*self.batch_size:(i+1)*self.batch_size] = gen_cap

                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, "{}/val/val.candidate.captions.pkl".format(self.data_path))
                    scores = evaluate(data_path= self.data_path, split='val', get_scores=True)
                    write_bleu(scores=scores, path=self.model_path, epoch=e)

                # save model's parameters
                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                    print "model-%s saved." %(e+1)

                sys.stdout.flush()


    def test(self, data, split='train', attention_visualization=True, save_sampled_captions=True, save_folder = 'plots', dynamic_image = False):
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
        if (not os.path.exists(save_folder)):
            os.makedirs(save_folder)


        features = data['features']
        if self.use_tag:
            tags = data['tags']

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=20)    # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            if self.use_tag:
                features_batch, tags_batch, image_files = sample_coco_minibatch_with_tags(data, self.batch_size)
                feed_dict = { self.model.features: features_batch, self.model.tags: tags_batch }
            else:
                features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
                feed_dict = { self.model.features: features_batch }
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)

            if attention_visualization:
                # just check 10 random samples
                for n in range(len(decoded)):
                    print "Sampled Caption: %s" %decoded[n]

                    # Plot original video frames
                    this_video = image_files[n]
                    # try:
                    videodata = skvideo.io.vread(str(this_video))
                    frame_count = videodata.shape[0]
                    frames_selected = np.arange(0, frame_count, int(np.ceil(frame_count/10.0))) # every couple of frames

                    if (dynamic_image):
                        frame_weight = 1.0 / len(frames_selected)

                        # get average image across frames
                        img = videodata[0]
                        img = skimage.transform.resize(img, (255, 255)) # inception original size is 229, 229
                        avg_image = np.zeros(img.shape)
                        for frame_pos in frames_selected:
                            img = videodata[frame_pos]
                            img = skimage.transform.resize(img, (255, 255))
                            avg_image = avg_image + frame_weight * img

                        plt.subplot(4, 5, 1)
                        plt.imshow(avg_image)
                        plt.axis('off')

                        # Plot images with attention weights
                        words = decoded[n].split(" ")
                        for t in range(len(words)):
                            if t > 18:
                                break
                            plt.subplot(4, 5, t+2)
                            plt.text(0, 1, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
                            plt.imshow(avg_image)
                            alp_curr = alps[n,t,:].reshape(8,8)
                            alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=32, sigma=20)
                            plt.imshow(alp_img, alpha=0.85)
                            plt.axis('off')
                        # plt.show()
                        plt.savefig('{}/sample{}.png'.format(save_folder, n))
                        plt.clf()

                    else:
                        T = len(decoded[n].split(" ")) + 1
                        for (i, frame_pos) in enumerate(frames_selected):
                            if (i > 10):
                                continue
                            print "plot frame:", frame_pos, "/", frame_count
                            img = videodata[frame_pos]
                            img = skimage.transform.resize(img, (255, 255)) # inception original size is 229, 229
                            plt.subplot(10, T, T*i + 1)
                            plt.imshow(img)
                            plt.axis('off')

                            # Plot images with attention weights
                            words = decoded[n].split(" ")
                            for t in range(len(words)):
                                if t > 18:
                                    break
                                plt.subplot(10, T, T*i + t+2)
                                print "plot at:", T*i + t+2, '/', 10*T
                                plt.text(0, 0, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor=(1, 1, 1, 0.0), fontsize=7)
                                plt.imshow(img)
                                alp_curr = alps[n,t,:].reshape(8,8)
                                alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=32, sigma=20)
                                plt.imshow(alp_img, alpha=0.85)
                                plt.axis('off')

                        plt.tight_layout(pad=0.3, w_pad=-2, h_pad=-1.6) # TODO: this issue?
                        plt.savefig('{}/sample{}.png'.format(save_folder, n), dpi=900)
                        plt.clf()
                        # matplotlib.rcdefaults()
                        plt.close('all')

                    # except:
                    #     print "video ", this_video, " unreadable"

            if save_sampled_captions:
                all_sam_cap = np.ndarray((features.shape[0], 20))
                num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
                for i in range(num_iter):
                    features_batch = features[i*self.batch_size:(i+1)*self.batch_size]
                    if self.use_tag:
                        tags_batch = tags[i*self.batch_size:(i+1)*self.batch_size]
                        feed_dict = { self.model.features: features_batch, self.model.tags: tags_batch }
                    else:
                        feed_dict = { self.model.features: features_batch }
                    all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)
                all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
                save_pickle(all_decoded, "%s/%s/%s.candidate.captions.pkl" %(self.data_path, split, split))

    def test_one_video(self, feature, this_video, attention_visualization=True, save_sampled_captions=True, save_folder = 'plots', dynamic_image = False, tag = None):
        '''
        Args:
            - data: dictionary with the following keys:
            - feature: feature vector of this one video
            - this_video: video filename of this given one test
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''
        video_name = this_video[max(this_video.rfind('/')+1, 0):this_video.rfind('.')]
        video_folder = this_video[:this_video.rfind('/') + 1] + video_name + '/' # the folder corresponding to this video name
        save_folder = video_folder + save_folder
        if (not os.path.exists(save_folder)):
            os.makedirs(save_folder)

        features = np.expand_dims(feature, axis = 0) # make one batch

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=20)    # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            features_batch = np.tile(features, (self.batch_size, 1, 1)) # hacky way to get around being squeezed
            print "features_batch.shape:", features_batch.shape

            if (not (tag is None)):
                tags = np.expand_dims(tag, axis = 0) # tags is (D, V)
                tags_batch = np.tile(tags, (self.batch_size, 1)) # hacky way to get around being squeezed
                feed_dict = { self.model.features: features_batch, self.model.tags: tags_batch }
            else:
                feed_dict = { self.model.features: features_batch }

            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)
            print "Caption for this video: %s" %decoded[0]

            if attention_visualization:

                # try:
                reader = skvideo.io.FFmpegReader(str(this_video))
                frame_count = reader.getShape()[0]
                print "frame_count:", frame_count
                frames_selected = np.arange(0, frame_count, int(np.ceil(frame_count/10.0))) # every couple of frames

                if (dynamic_image):
                    frame_weight = 1.0 / len(frames_selected)
                    avg_image = np.zeros(tuple(reader.getShape()[1:]))

                    for cur_frame in range(0, frame_count):
                        try:
                            img = reader.nextFrame().next()
                            if (cur_frame in frames_selected):
                                img = skimage.transform.resize(img, (255, 255))
                                avg_image = avg_image + frame_weight * img
                        except:
                            print "frame", cur_frame, " can not be read"

                    plt.subplot(4, 5, 1)
                    plt.imshow(avg_image)
                    plt.axis('off')

                    # Plot images with attention weights
                    words = decoded[0].split(" ")
                    for t in range(len(words)):
                        if t > 18:
                            break
                        plt.subplot(4, 5, t+2)
                        plt.text(0, 1, '%s(%.2f)'%(words[t], bts[0,t]) , color='black', backgroundcolor='white', fontsize=8)
                        plt.imshow(avg_image)
                        alp_curr = alps[0,t,:].reshape(8,8)
                        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=32, sigma=20)
                        plt.imshow(alp_img, alpha=0.85)
                        plt.axis('off')
                    # plt.show()
                    plt.savefig('{}/{}.png'.format(save_folder, video_name))
                    plt.clf()

                else:
                    i = 0
                    for cur_frame in range(0, frame_count):
                        try:
                            img = reader.nextFrame().next()
                            if (cur_frame in frames_selected):
                                T = len(decoded[0].split(" ")) + 1
                                print "plot frame:", cur_frame, "/", frame_count
                                img = skimage.transform.resize(img, (255, 255)) # inception original size is 229, 229
                                # plot maximum 10 rows
                                if i >= 10:
                                    continue
                                plt.subplot(10, T, T*i + 1)
                                plt.imshow(img)
                                plt.axis('off')

                                # Plot images with attention weights
                                words = decoded[0].split(" ")
                                for t in range(len(words)):
                                    if t > 18:
                                        break
                                    ax = plt.subplot(10, T, T*i + t+2)
                                    plt.text(0, 0, '%s(%.2f)'%(words[t], bts[0,t]) , color='black', backgroundcolor=(1, 1, 1, 0.0), fontsize=7)
                                    plt.imshow(img)
                                    alp_curr = alps[0,t,:].reshape(8,8)
                                    alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=32, sigma=20)
                                    plt.imshow(alp_img, alpha=0.85)
                                    plt.axis('off')
                                i += 1
                        except:
                            print "frame ", cur_frame, " can not be read"
                    # plt.show()
                    plt.tight_layout(pad=0.3, w_pad=-2, h_pad=-1.6)
                    plt.savefig('{}/{}.png'.format(save_folder, video_name), dpi=900)
                    plt.clf()
                # except:
                #     print "video ", this_video, " unreadable"

            if save_sampled_captions:
                save_pickle(decoded, video_folder + "one.candidate.caption.pkl")

        return decoded[0]


    def test_given_data(self, data, save_folder = None):
        '''
        Args:
            - data: dictionary with the following keys:
            - save_folder: place to save the captions, if not
        '''

        features = data['features']
        if self.use_tag:
            tags = data['tags']

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=20) # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            # get test captions
            all_sam_cap = np.ndarray((features.shape[0], 20))
            num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
            for i in range(num_iter):
                features_batch = features[i*self.batch_size:(i+1)*self.batch_size]
                if self.use_tag:
                    tags_batch = tags[i*self.batch_size:(i+1)*self.batch_size]
                    feed_dict = { self.model.features: features_batch, self.model.tags: tags_batch }
                else:
                    feed_dict = { self.model.features: features_batch }
                all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)
            all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)

            # check if need to save
            if not save_folder is None:
                # create folder
                if (not os.path.exists(save_folder)):
                    os.makedirs(save_folder)
                if (save_folder[-1] == '/'):
                    save_folder = save_folder[:-1]
                save_pickle(all_decoded, "%s/candidate.captions.pkl" %(save_folder))

            return all_decoded