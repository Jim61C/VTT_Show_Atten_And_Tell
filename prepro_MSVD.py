from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json
import cPickle as pickle
import config

APPEND_MFCC = True

def _process_caption_data(caption_file, video_dir, max_length):
    with open(caption_file) as f:
        caption_lines = f.readlines()

    video2id = pickle.load(open(config.MSVD_VIDEO2ID, 'r'))

    data = []
    for i, line in enumerate(caption_lines):
        entries = []
        for _ in xrange(7):
            idx = line.find(',')
            entries.append(line[0:idx])
            line = line[idx+1:]
        entries.append(line.strip())

        if entries[6] != 'English':
            continue

        annotation = {}
        file_name = entries[0] + '_' + entries[1] + '_' + entries[2]
        if file_name not in video2id:
            continue

        annotation['image_id'] = video2id[file_name]
        annotation['file_name'] = os.path.join(video_dir, file_name + '.avi')
        annotation['caption'] = entries[-1]
        data.append(annotation)
    
    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data)
    caption_data.sort_values(by='image_id', inplace=True)
    caption_data = caption_data.reset_index(drop=True)
    
    del_idx = []
    for i, caption in enumerate(caption_data['caption']):
        caption = caption.replace('.','').replace(',','').replace("'","").replace('"','')
        caption = caption.replace('&','and').replace('(','').replace(")","").replace('-',' ')
        caption = " ".join(caption.split())  # replace multiple spaces
        
        caption_data.set_value(i, 'caption', caption.lower())
        if len(caption.split(" ")) > max_length:
            del_idx.append(i)
    
    # delete captions if size is larger than max_length
    print "The number of captions before deletion: %d" %len(caption_data)
    caption_data = caption_data.drop(caption_data.index[del_idx])
    caption_data = caption_data.reset_index(drop=True)
    print "The number of captions after deletion: %d" %len(caption_data)
    return caption_data


def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(annotations['caption']):
        words = caption.split(' ') # caption contrains only lower-case words
        for w in words:
            counter[w] +=1
        
        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print "Max length of caption: ", max_len
    return word_to_idx


def _build_caption_vector(annotations, word_to_idx, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)   

    for i, caption in enumerate(annotations['caption']):
        words = caption.split(" ") # caption contrains only lower-case words
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])
        
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>']) 
        
        captions[i, :] = np.asarray(cap_vec)
    print "Finished building caption vectors"
    return captions


def _build_file_names(annotations):
    image_file_names = []
    id_to_idx = {}
    idx = 0
    video_ids = annotations['image_id']
    file_names = annotations['file_name']
    for video_id, file_name in zip(video_ids, file_names):
        if not video_id in id_to_idx:
            id_to_idx[video_id] = idx
            image_file_names.append(file_name)
            idx += 1

    file_names = np.asarray(image_file_names)
    return file_names, id_to_idx


def _build_video_idxs(annotations, id_to_idx):
    video_idxs = np.ndarray(len(annotations), dtype=np.int32)
    video_ids = annotations['image_id']
    for i, video_id in enumerate(video_ids):
        video_idxs[i] = id_to_idx[video_id]
    return video_idxs


def main():
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.  
    max_length = 15
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 5

    this_dataset = 'data_MSVD_mfcc'

    # whole dataset 
    dataset = _process_caption_data(caption_file= config.MSVD_CAPTION_FILE,
                                         video_dir= config.MSVD_VIDEO_DIR,
                                         max_length=max_length)
    print 'Finished processing caption data'

    # train split: 0:1200, val split: 1200:1300, test split: 1300:
    train_dataset = dataset[dataset['image_id'] < 1200]
    val_dataset   = dataset[(dataset['image_id'] >= 1200) & (dataset['image_id'] < 1300)].reset_index(drop=True)
    test_dataset  = dataset[dataset['image_id'] >= 1300].reset_index(drop=True)
    
    save_pickle(train_dataset, '%s/train/train.annotations.pkl' % this_dataset)
    save_pickle(val_dataset, '%s/val/val.annotations.pkl' % this_dataset)
    save_pickle(test_dataset, '%s/test/test.annotations.pkl' % this_dataset)

    all_features = np.load(config.MSVD_FEATURE_INCEPTION)
    all_features = np.expand_dims(all_features, axis = 1)

    for split in ['train', 'val', 'test']:
        annotations = load_pickle('./%s/%s/%s.annotations.pkl' % (this_dataset, split, split))

        print "len(train annotations): ", len(annotations)

        if split == 'train':
            #word_to_idx = _build_vocab(annotations=annotations, threshold=word_count_threshold)
            word_to_idx = load_pickle('./word_to_idx_all.pkl')
            save_pickle(word_to_idx, './%s/%s/word_to_idx.pkl' % (this_dataset, split))
        
        captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
        save_pickle(captions, './%s/%s/%s.captions.pkl' % (this_dataset, split, split))

        file_names, id_to_idx = _build_file_names(annotations)
        save_pickle(file_names, './%s/%s/%s.file.names.pkl' % (this_dataset, split, split))

        video_idxs = _build_video_idxs(annotations, id_to_idx)
        save_pickle(video_idxs, './%s/%s/%s.image.idxs.pkl' % (this_dataset, split, split))

        # prepare reference captions to compute bleu scores later
        if (APPEND_MFCC):
            mfcc_feature_total = np.load(config.MSVD_FEATURE_AUDIO)
            video2id = load_pickle('/home/ubuntu/features/msvd/video2int.pkl')

        video_ids = {}
        feature_to_captions = {}
        features = []
        i = -1
        for caption, video_id, video_file in zip(annotations['caption'], annotations['image_id'], annotations['file_name']):
            if not video_id in video_ids:
                video_ids[video_id] = 0
                i += 1
                feature_to_captions[i] = []
                # append feature
                this_video_feature = all_features[int(video_id)]
                # append MFCC if needed
                if (APPEND_MFCC):
                    this_video_mfcc_feature = mfcc_feature_total[video2id[video_file[video_file.rfind('/')+1:video_file.find('.avi')]]]
                    assert(video_id == video2id[video_file[video_file.rfind('/')+1:video_file.find('.avi')]]), \
                    "video_id inconsistent for " + video_file[video_file.rfind('/')+1:video_file.find('.avi')]
                    this_video_mfcc_feature = np.tile(this_video_mfcc_feature, (config.SPATIAL_DIM, 1))
                    print "this_video_mfcc_feature.shape:", this_video_mfcc_feature.shape
                    this_video_feature = np.concatenate((this_video_feature, this_video_mfcc_feature), axis = 1)
                    print "this_video_feature.shape:", this_video_feature.shape
                features.append(this_video_feature)
                print "{} feature appended :".format(split), video_file

            feature_to_captions[i].append(caption.lower() + ' .')

        hickle.dump(np.asarray(features), './%s/%s/%s.features.hkl' % (this_dataset, split, split))
        save_pickle(feature_to_captions, './%s/%s/%s.references.pkl' % (this_dataset, split, split))
        print "Finished building %s caption dataset" %split



if __name__ == "__main__":
    main()