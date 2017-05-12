import cPickle as pickle
import hickle
import numpy as np
import argparse


if __name__ == "__main__":
    word2idx = pickle.load(open('word_to_idx_all.pkl', 'rb'))
    for split in ['train', 'val', 'test']:
        print 'loading {} tags'.format(split)
        tags = pickle.load(open('tags.{}.MSRVTT.pickle'.format(split), 'rb'))
        tag_array = []
        for k in sorted(tags.keys()):
            tag_vec = np.zeros(shape = (len(word2idx)))
            for w in tags[k]:
                if w in word2idx:
                    tag_vec[word2idx[w]] = 1
            tag_array.append(tag_vec)
        tag_array = np.array(tag_array)
        print tag_array
        print tag_array.shape
        print tag_array.sum(1)
        print tag_array.sum(1).shape
        print 'saving {} tag vectors'.format(split)
        hickle.dump(tag_array, open('{}.tag.vectors.hkl'.format(split), 'w'))