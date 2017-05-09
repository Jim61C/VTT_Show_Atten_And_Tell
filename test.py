import matplotlib
# matplotlib.use('TkAgg')  # for visualisation + writing
# matplotlib.use('WX')  # for visualisation + writing
matplotlib.use('Agg')  # for just writing
import matplotlib.pyplot as plt
import cPickle as pickle
import tensorflow as tf
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
from core.bleu import evaluate


plt.rcParams['figure.figsize'] = (8.0, 6.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'



def main():
	# load val dataset to print out bleu scores every epoch
	val_data = load_coco_data(data_path='./data_MSRVTT', split='val')
	# test_data = load_coco_data(data_path='./data_MSRVTT', split='test')

	with open('./data_MSRVTT/train/word_to_idx.pkl') as f:
		word_to_idx = pickle.load(f)

	model = CaptionGenerator(word_to_idx, dim_feature=[64, 2048], dim_embed=512,
									   dim_hidden=1024, n_time_step=16, prev2out=True, 
												 ctx2out=True, alpha_c=1.0, selector=True, dropout=True, device_id = '/gpu:0')

	# Test, put data as dummy (here just use val_data)
	solver = CaptioningSolver(model, val_data, val_data, n_epochs=20, batch_size=98, update_rule='adam',
										  learning_rate=0.001, print_every=1000, save_every=1, image_path='./image/',
									pretrained_model=None, model_path='model/lstm/', test_model='model/lstm/model-11',
									 print_bleu=True, log_path='log/')

	# Test, save produced captions
	solver.test(val_data, split='val', attention_visualization=True, save_sampled_captions = True, save_folder = 'plots/val')
	# tf.get_variable_scope().reuse_variables()
	# solver.test(test_data, split='test', attention_visualization=True, save_sampled_captions = True, save_folder = 'plots/test')

	# Evaluation
	print "Evaluation, validation set..."
	evaluate(data_path='./data_MSRVTT', split='val')

	# print "Evaluation, test set..."
	# evaluate(data_path='./data_MSRVTT', split='test')


if __name__ == "__main__":
	main()
