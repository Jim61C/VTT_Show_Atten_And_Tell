import os

assert 'OPENIMAGES_PATH' in os.environ, 'please first set env_var OPENIMAGES_PATH as the absolute path of OPENIMAGES'
OPENIMAGES_PATH = os.environ['OPENIMAGES_PATH']

check_point = OPENIMAGES_PATH + '/data/2016_08/model.ckpt'
label_map = OPENIMAGES_PATH + '/data/2016_08/labelmap.txt'
csv_dict = OPENIMAGES_PATH + '/dict.csv'
image_size = 299
num_classes = 6012
num_tags = 5