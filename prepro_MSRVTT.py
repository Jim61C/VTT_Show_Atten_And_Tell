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
import cPickle


def _process_caption_data(caption_file, video_dir, video_exist, max_length):
    with open(caption_file) as f:
        caption_data = json.load(f)

    video_exist_dict = None
    if (video_exist != 'dummy/'):
        video_exist_dict = cPickle.load(open(video_exist, 'rb'))

    # id_to_filename is a dictionary such as {video_id: filename]}
    id_to_filename = {video['id']: video['video_id'] for video in caption_data['videos']}
    filename_to_id = {id_to_filename[id]:id for id in id_to_filename}

    # data is a list of dictionary which contains 'caption', 'file_name' and 'video_id' as key.
    data = []
    for annotation in caption_data['sentences']:
        #video_id = annotation['video_id']
        annotation['file_name'] = os.path.join(video_dir, annotation['video_id']) + '.mp4'
        annotation['image_id'] = int(annotation['video_id'][5:])
        if (video_exist_dict is None or video_exist_dict[annotation['video_id'][5:]]):
            data += [annotation]

    # convert to pandas dataframe (for later visualization or debugging)
    caption_data = pd.DataFrame.from_dict(data)
    del caption_data['sen_id']
    del caption_data['video_id']
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
    ""
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.
    max_length = 15
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 5

    caption_file = 'data_MSRTT/annotations/train_val_videodatainfo.json'
    video_dir = '/mnt/sdb1/yxing1/show_atten_tell/videos/'

    # train split: 0:6512, val split: 6513:7009
    train_val_dataset = _process_caption_data(caption_file='data_MSRVTT/annotations/train_val_videodatainfo.json',
                                              video_dir=video_dir,
                                              video_exist='dummy/',
                                              max_length=max_length)
    train_dataset = train_val_dataset[train_val_dataset['image_id'] <= 6512]
    val_dataset = train_val_dataset[train_val_dataset['image_id'] > 6512].reset_index(drop=True)

    # test split 7010:9999
    test_dataset = _process_caption_data(caption_file='data_MSRVTT/annotations/test_videodatainfo.json',
                                         video_dir=video_dir,
                                         video_exist='dummy/',
                                         max_length=max_length)
    print 'Finished processing caption data'

    save_pickle(train_dataset, 'data_MSRVTT/train/train.annotations.pkl')
    save_pickle(val_dataset, 'data_MSRVTT/val/val.annotations.pkl')
    save_pickle(test_dataset, 'data_MSRVTT/test/test.annotations.pkl')


    feature_split_start_idx = {
    'train': 0, 
    'val': 6513,
    'test': 7010
    }
    for split in ['train', 'val', 'test']:
        annotations = load_pickle('./data_MSRVTT/%s/%s.annotations.pkl' % (split, split))

        if split == 'train':
            #word_to_idx = _build_vocab(annotations=annotations, threshold=word_count_threshold)
            word_to_idx = load_pickle('./word_to_idx_all.pkl')
            save_pickle(word_to_idx, './data_MSRVTT/%s/word_to_idx.pkl' % split)

        captions = _build_caption_vector(annotations=annotations, word_to_idx=word_to_idx, max_length=max_length)
        save_pickle(captions, './data_MSRVTT/%s/%s.captions.pkl' % (split, split))

        file_names, id_to_idx = _build_file_names(annotations)
        save_pickle(file_names, './data_MSRVTT/%s/%s.file.names.pkl' % (split, split))

        video_idxs = _build_video_idxs(annotations, id_to_idx)
        save_pickle(video_idxs, './data_MSRVTT/%s/%s.image.idxs.pkl' % (split, split))

        # prepare reference captions to compute bleu scores later
        video_ids = set()
        feature_to_captions = {}
        features = [] # needed as some video clip might not be valid and these clip's features are dummy, thus should be removed from original feature array
        
        full_feature_data = None
        full_feature_path = './data_MSRVTT/%s/%s.features.full.hkl' % (split, split)
        if (os.path.exists(full_feature_path)):
            full_feature_data = hickle.load(full_feature_path)

        i = -1
        for caption, video_id in zip(annotations['caption'], annotations['image_id']):
            if not video_id in video_ids:
                video_ids.add(video_id)
                i += 1
                feature_to_captions[i] = []
                if (not full_feature_data is None):
                    features.append(full_feature_data[video_id - feature_split_start_idx[split]])

            feature_to_captions[i].append(caption.lower() + ' .')
        save_pickle(feature_to_captions, './data_MSRVTT/%s/%s.references.pkl' % (split, split))
        if (len(features) > 0):
            hickle.dump(np.asarray(features), './data_MSRVTT/%s/%s.features.hkl' % (split, split))
        print "Finished building %s caption dataset" %split

if __name__ == "__main__":
    main()
