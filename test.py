import matplotlib
# matplotlib.use('TkAgg')  # for visualisation + writing
# matplotlib.use('WX')  # for visualisation + writing
matplotlib.use('Agg')  # for just writing
import matplotlib.pyplot as plt
import cPickle as pickle
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
from core.utils import save_pickle
from core.bleu import evaluate
import tensorflow as tf
import sys
import os
import csv
from feature_extractor.extract_inception_v3 import FeatureExtractor
from tag_extractor.extract_tag import TagExtractor
import config
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8") # to get rid of ASCII decoding issue


plt.rcParams['figure.figsize'] = (8.0, 6.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def main():
	# load val dataset to print out bleu scores every epoch
	val_data = load_coco_data(data_path=config.DATASET, split='val')
	test_data = load_coco_data(data_path=config.DATASET, split='test')

	with open('{}/train/word_to_idx.pkl'.format(config.DATASET)) as f:
		word_to_idx = pickle.load(f)

	model = CaptionGenerator(word_to_idx, dim_feature=[config.SPATIAL_DIM, 2048], dim_embed=512,
									   dim_hidden=1024, n_time_step=16, prev2out=True,
												 ctx2out=True, alpha_c=1.0, selector=True, dropout=True, device_id = '/gpu:0')

	# Test, put data as dummy (here just use val_data)
	solver = CaptioningSolver(model, val_data, val_data, n_epochs=20, batch_size=98, update_rule='adam',
										  learning_rate=0.001, print_every=1000, save_every=1, image_path='./image/',
									pretrained_model=None, model_path=config.MODEL_PATH, test_model='{}/model-7'.format(config.MODEL_PATH),
									 print_bleu=True, log_path='log/', data_path=config.DATASET)


	# Test, save produced captions
	solver.test(val_data, split='val', attention_visualization=True, save_sampled_captions = True, save_folder = 'plots_{}/val'.format(config.DATASET_SUFFIX))
	tf.get_variable_scope().reuse_variables()
	solver.test(test_data, split='test', attention_visualization=True, save_sampled_captions = True, save_folder = 'plots_{}/test'.format(config.DATASET_SUFFIX))

	# Evaluation
	print "Evaluation, validation set..."
	evaluate(data_path=config.DATASET, split='val')

	print "Evaluation, test set..."
	evaluate(data_path=config.DATASET, split='test')

	print "End of Test!"

def test_to_csv():
	this_split = sys.argv[2]
	print "loading split ", this_split, '...'
	# load val dataset to print out bleu scores every epoch
	data = load_coco_data(data_path=config.DATASET, split=this_split)

	with open('{}/train/word_to_idx.pkl'.format(config.DATASET)) as f:
		word_to_idx = pickle.load(f)

	model = CaptionGenerator(word_to_idx, dim_feature=[config.SPATIAL_DIM, 2048], dim_embed=512,
									   dim_hidden=1024, n_time_step=16, prev2out=True,
												 ctx2out=True, alpha_c=1.0, selector=True, dropout=True, device_id = '/gpu:0')
	# Test, put data as dummy
	solver = CaptioningSolver(model, data, data, n_epochs=20, batch_size=98, update_rule='adam',
									  learning_rate=0.001, print_every=1000, save_every=1, image_path='./image/',
								pretrained_model=None, model_path=config.MODEL_PATH, test_model='{}/model-7'.format(config.MODEL_PATH),
								 print_bleu=True, log_path='log/', data_path=config.DATASET)
	scores_save = {
	'Bleu_1': [],
	'Bleu_2': [],
	'Bleu_3': [],
	'Bleu_4': [],
	'METEOR': [],
	'ROUGE_L': [],
	'CIDEr': []
	}

	# saved models are index 1 based
	for i in range(1, 21):
		model_id = str(i)
		print "evaluate for model-", model_id
		solver.test_model='{}/model-{}'.format(config.MODEL_PATH, model_id)

		if i > 1:
			tf.get_variable_scope().reuse_variables()
		solver.test(data, split=this_split, attention_visualization=False, save_sampled_captions = True, save_folder = 'plots_{}/{}'.format(config.DATASET_SUFFIX, this_split))
		final_scores = evaluate(data_path=config.DATASET, split=this_split, get_scores=True)

		for (metric, scores) in scores_save.iteritems():
			scores.append(final_scores[metric])


	for i in range(1, len(scores_save.keys())):
		assert (len(scores_save[scores_save.keys()[0]]) == len(scores_save[scores_save.keys()[i]])), \
		'metric ' + scores_save.keys()[i] + " do not have the same amount of data"

	if(not os.path.exists('results_{}/'.format(config.DATASET_SUFFIX))):
		os.makedirs('results_{}/'.format(config.DATASET_SUFFIX))

	with open('results_{}/{}.csv'.format(config.DATASET_SUFFIX, this_split), 'wb') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=scores_save.keys())

		writer.writeheader()
		for i in range(0, len(scores_save['METEOR'])):
			row_dict = {}
			for (metric, scores) in scores_save.iteritems():
				row_dict[metric] = scores[i]
			writer.writerow(row_dict)


def test_one():
	# use tag from OpenImage
	use_tag = True

	video_file = sys.argv[2]
	video_name = video_file[max(video_file.rfind('/')+1, 0):video_file.rfind('.')]
	
	# get feature
	feature_path = config.TEST_ONE_VIDEO_FEATURE_TEMPLATE.format(video_name)
	this_feature = None
	if (os.path.exists(feature_path)):
		this_feature = pickle.load(open(feature_path, 'rb'))
		print "feature already extracted for ", video_file
	else:
		extactor = FeatureExtractor('mixed_10/join:0')
		avg_mided_10_feature_shape = (8,8,2048)

		mixed_10_join_feature = extactor.extract_feature_video(video_file, \
			None, None, avg_mided_10_feature_shape)
		if (not mixed_10_join_feature is None):
			this_feature = mixed_10_join_feature.reshape(-1, 2048)

		save_pickle(this_feature, feature_path)

	if (this_feature is None):
		raise ValueError('The Given Video File ' + video_file + " gets no feature extracted!")

	# get tag
	tag_path = config.TEST_ONE_VIDEO_TAG_TEMPLATE.format(video_name)
	this_tag = None
	if (os.path.exists(tag_path)):
		this_tag = pickle.load(open(tag_path, 'rb'))
		print "tag already extracted for ", video_file
	else:
		# extract tag for this video_file
		tag_extractor = TagExtractor()
		this_tag_list = tag_extractor.extract_tags_one_video(video_file, num_frames = 10)
		print "this_tag_list: ", this_tag_list
		this_tag = tag_extractor.convertTagToVector(this_tag_list)
		print "this_tag: ", this_tag
		# save it
		save_pickle(this_tag, tag_path)

	if (this_tag is None):
		raise ValueError("The video " + video_file + " gets no tag extracted!")

	# set up test model
	with open('{}/train/word_to_idx.pkl'.format(config.DATASET)) as f:
		word_to_idx = pickle.load(f)

	model = CaptionGenerator(word_to_idx, dim_feature=[config.SPATIAL_DIM, 2048], dim_embed=512,
									   dim_hidden=1024, n_time_step=16, prev2out=True,
												 ctx2out=True, alpha_c=1.0, selector=True, dropout=True, use_tag = use_tag, device_id = '/gpu:0')

	# Test, put data as dummy (not used)
	solver = CaptioningSolver(model, [], [], n_epochs=20, batch_size=2, update_rule='adam',
										  learning_rate=0.001, print_every=1000, save_every=1, image_path='./image/',
									pretrained_model=None, model_path=config.MODEL_PATH, test_model='{}/model-7'.format(config.MODEL_PATH),
									 print_bleu=True, use_tag = use_tag, log_path='log/')


	# Test, save produced captions
	solver.test_one_video(this_feature, video_file, tag = this_tag, attention_visualization=True, save_sampled_captions = True, save_folder = 'plots_test_one/', dynamic_image = False)



if __name__ == "__main__":
	if (len(sys.argv) < 2):
		print "Usage: python {} option [split]".format(sys.argv[0])
		print "option -- E.g., csv/visualise/one"
		print "[split] -- in case of 'csv' option, E.g., test, then test.csv will be saved"

		print "\nUsage: python {} one video_file".format(sys.argv[0])
		print "video_file -- the path to the one video to test on"
		exit()
	if (sys.argv[1] == 'csv'):
		test_to_csv()
	elif (sys.argv[1] == 'one'):
		test_one()
	else:
		main()
