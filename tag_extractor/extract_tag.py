from scipy import ndimage
from collections import Counter

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import sys
import json
import cPickle as pickle

import skvideo.io
import argparse
import TAGConfig

from tensorflow.contrib.slim.python.slim.nets import inception

slim = tf.contrib.slim


# os.path.join('/home/ubuntu/openimages')
#from tools.classify import *

class TagExtractor(object):
    def __init__(self):
        self.construct_graph()
        self.word2idx = pickle.load(open('word_to_idx_all.pkl', 'rb'))
        return

    def construct_graph(self):
        global TAGConfig
        self.g = tf.Graph()
        with self.g.as_default():
            # create input image placeholder
            input_image = tf.placeholder(
                dtype = tf.float32,
                shape = (None, None, None, 3)
            )
            resized_image = tf.image.resize_bilinear(
                input_image,
                [TAGConfig.image_size, TAGConfig.image_size],
                align_corners=False)
            processed_image = tf.subtract(tf.multiply(resized_image, 1.0/127.5), 1.0)
            # create inception_v3 model
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                logits, end_points = inception.inception_v3(
                    processed_image, num_classes=TAGConfig.num_classes, is_training=False)

            predictions = end_points['multi_predictions'] = tf.nn.sigmoid(
                logits, name='multi_predictions')
            saver = tf.train.Saver()
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config = config)
            # load pretrain model
            saver.restore(sess, TAGConfig.check_point)

            # extract tags
            labelmap, label_dict = LoadLabelMaps(TAGConfig.num_classes, TAGConfig.label_map, TAGConfig.csv_dict)

            self.input_image = input_image
            self.predictions = predictions
            self.labelmap = labelmap
            self.label_dict = label_dict
            self.sess = sess

        

    def extract_tags_one_video(self, video_path, num_frames = 10):
        global TAGConfig
        tag_list = {}
        with self.g.as_default():
            frames = skvideo.io.vread(video_path)
            frame_step = frames.shape[0] / num_frames
            for i in xrange(num_frames):
                idx = np.random.choice(np.arange(round(frame_step * i), round(frame_step * (i+1)), dtype = np.int32))
                frame = frames[idx, :, :, :]
                predictions_eval = np.squeeze(
                    self.sess.run(self.predictions, {self.input_image: frame[np.newaxis, :, :, :]}))
                top_k = predictions_eval.argsort()[-TAGConfig.num_tags:][::-1]
                for idx in top_k:
                    mid = self.labelmap[idx]
                    display_name = self.label_dict.get(mid, 'unknown')
                    score = predictions_eval[idx]
                    #print('{}: {} (score = {:.2f})'.format(idx, display_name, score))
                    for word in display_name.split():
                        tag_list[word.strip()] = 0
        return sorted(tag_list.keys())

    def extract_tags(self, data_dict, num_frames = 10):
        tags = {}
        for k in sorted(data_dict.keys()):
            sorted_tag_list = self.extract_tags_one_video(str(data_dict[k]), num_frames)
            tags[k] = sorted_tag_list
            print k
            print tags[k]
        return tags

    def convertTagToVector(self, tag_list):
        tag_vec = np.zeros(shape = (len(self.word2idx)))
        for w in tag_list:
            if w in self.word2idx:
                tag_vec[self.word2idx[w]] = 1

        return tag_vec


def load_dataset(caption_file, video_dir):
    with open(caption_file) as f:
        caption_data = json.load(f)

    # id_to_filename is a dictionary such as {video_id: filename]}
    id_to_filename = {video['id']: video['video_id'] for video in caption_data['videos']}
    
    for key in id_to_filename:
        id_to_filename[key] = os.path.join(video_dir, id_to_filename[key]) + '.mp4'
        #print id_to_filename[key]
    return id_to_filename




def LoadLabelMaps(num_classes, labelmap_path, dict_path):
    """Load index->mid and mid->display name maps.

    Args:
        labelmap_path: path to the file with the list of mids, describing predictions.
        dict_path: path to the dict.csv that translates from mids to display names.
    Returns:
        labelmap: an index to mid list
        label_dict: mid to display name dictionary
    """
    labelmap = [line.rstrip() for line in tf.gfile.GFile(labelmap_path).readlines()]
    if len(labelmap) != num_classes:
        tf.logging.fatal(
            "Label map loaded from {} contains {} lines while the number of classes is {}".format(
                labelmap_path, len(labelmap), num_classes))
        sys.exit(1)

    label_dict = {}
    for line in tf.gfile.GFile(dict_path).readlines():
        words = [word.strip(' "\n') for word in line.split(',', 1)]
        label_dict[words[0]] = words[1]

    return labelmap, label_dict


if __name__ == "__main__":
    train_val_dict = load_dataset(
        caption_file='../train_val_videodatainfo.json',
        video_dir='../videos')
    test_dict = load_dataset(
        caption_file='../test_videodatainfo.json',
        video_dir='../videos')

    extactor = TagExtractor()

    train_val_tags = extactor.extract_tags(train_val_dict)
    
    train_tags = {}
    for i in xrange(0, 6513):
        train_tags[i] = train_val_tags[i]
    # pickle.dump(train_tags, open("tags.train.MSRVTT.pickle", 'wb'))
    
    val_tags = {}
    for i in xrange(6513, 7010):
        val_tags[i] = train_val_tags[i]
    # pickle.dump(val_tags, open("tags.val.MSRVTT.pickle", 'wb'))
    
    test_tags = extactor.extract_tags(test_dict)
    # pickle.dump(test_tags, open("tags.test.MSRVTT.pickle", 'wb'))
    