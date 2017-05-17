# DATASET = './data_MSRVTT'
# DATASET_SUFFIX = 'MSRVTT'

# DATASET = './data_MSRVTT_mfcc'
# DATASET_SUFFIX = 'MSRVTT_mfcc'

DATASET = './data_MSVD'
DATASET_SUFFIX = 'MSVD'

# DATASET = './data_MSVD_mfcc'
# DATASET_SUFFIX = 'MSVD_mfcc'

# DATASET = './data_MSVD_pooled'
# DATASET_SUFFIX = 'MSVD_pooled'

# DATASET = './data_MSVD_pooled_mfcc'
# DATASET_SUFFIX = 'MSVD_pooled_mfcc'

SPATIAL_DIM = 64 # spatial locations L, use 1 for show and tell with no attention
FEAT_DIM = 2048
# FEAT_DIM = 2672

# MODEL_PATH = './model/show_attend_and_tell/mfcc_tag'
# MODEL_PATH = './model/show_attend_and_tell/tag' # do not put / at the end
# MODEL_PATH = './model/show_attend_and_tell/mfcc_tag_MSVD'
# MODEL_PATH = './model/show_attend_and_tell/tag_MSVD'
# MODEL_PATH = './model/show_attend_and_tell/mfcc_MSVD'
MODEL_PATH = './model/show_attend_and_tell/MSVD'
# MODEL_PATH = './model/show_and_tell/mfcc_tag_MSVD'
# MODEL_PATH = './model/show_and_tell/mfcc_MSVD'
# MODEL_PATH = './model/show_and_tell/tag_MSVD'
# MODEL_PATH = './model/show_and_tell/MSVD'

################################ MSRVTT ################################
VIDEO_DIR = '/home/ubuntu/video_data/MSRVTT/videos/'

TRAIN_CAPTION_FILE = DATASET + '/annotations/train_val_videodatainfo.json'

TEST_CAPTION_FILE = DATASET + '/annotations/test_videodatainfo.json'



################################ MSVD ################################
MSVD_VIDEO_DIR = '/home/ubuntu/video_data/MSVD/YouTubeClips' 

MSVD_CAPTION_FILE = DATASET + '/annotations/MSR_Video_Description_Corpus.csv'

MSVD_VIDEO2ID = DATASET + '/annotations/video2int.pkl'

MSVD_FEATURE_PATH_TEMPLATE = '/home/ubuntu/video_data/MSVD/mixed_10_join_feature_dict/{}/mixed_10_join_feature.pkl'

MSVD_MFCC_FEATURE_PATH = '/home/ubuntu/features/msvd/mfccfv.npy'



################################ Test One Video ################################
TEST_ONE_VIDEO_FEATURE_TEMPLATE = '/home/ubuntu/test_videos/{}/mixed_10_join_feature_dict/mixed_10_join_feature.pkl'
TEST_ONE_VIDEO_TAG_TEMPLATE = '/home/ubuntu/test_videos/{}/tag_vector.pkl'