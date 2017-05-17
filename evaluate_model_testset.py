
# coding: utf-8

# In[1]:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle as pickle
import tensorflow as tf
from core_scratch.solver import CaptioningSolver
from core_scratch.lstm_gan_model import CaptionGenerator
from core_scratch.utils import load_coco_data
from core_scratch.bleu import evaluate

#get_ipython().magic(u'matplotlib inline')
#plt.rcParams['figure.figsize'] = (8.0, 6.0)  # set default size of plots
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')

dataset = 'msrvtt_gan_scratch' #['mscoco', 'msrvtt', 'mscoco-msrvtt', 'vtt', 'msvd', 'msrvtt_mfcc']
split = 'test' #['val', 'test']

if dataset=='mscoco':
	data_path = './data'
	test_model = './model/show_and_tell/lstm/model-'
	num_model = 20
elif dataset=='msrvtt':
	data_path = './data_MSRVTT'
	test_model = '../show-attend-and-tell-MSRVTT_scratch/model/lstm_MSRVTT_from_scratch/model-'
	num_model = 40
elif dataset=='mscoco-msrvtt':
	data_path = './data_MSRVTT'
	test_model = './model/show_and_tell/lstm_finetune_MSRVTT/model-'
	num_model = 40
elif dataset=='vtt':
        data_path = './data_trecvid_2016'
        test_model = './model/show_and_tell/lstm_finetune_MSRVTT/model-'
        num_model = 40
elif dataset=='msrvtt_mfcc':
        data_path = './data_MSRVTT_mfcc'
        test_model = './model/show_and_tell/lstm_MSRVTT_mfcc/model-'
        num_model = 20
elif dataset=='msrvtt_tag':
    	data_path = './data_MSRVTT'
	test_model = './model/show_and_tell/lstm_MSRVTT_tag/model-'
	num_model = 20
elif dataset=='msrvtt_mfcc_tag':
    	data_path = './data_MSRVTT_mfcc'
	test_model = './model/show_and_tell/lstm_MSRVTT_mfcc_tag/model-'
	num_model = 20
elif dataset=='msrvtt_gan_scratch':
    	data_path = './data_MSRVTT'
	test_model = './model/show_and_tell/lstm_gan_scratch/model-'
	num_model = 40


data = load_coco_data(data_path=data_path, split='val')
with open(data_path + '/train/word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)

model = CaptionGenerator(word_to_idx, dim_feature=[1, 2048], dim_embed=512,
                         dim_hidden=1024, n_time_step=16, prev2out=True,
                         ctx2out=False, alpha_c=0.0, selector=False, dropout=True)

output = "Bleu_4,METEOR,CIDEr,SUM\n"
for i in range(6,7): #,num_model+1):
	solver = CaptioningSolver(model, data, data, n_epochs=15, batch_size=128, update_rule='adam',
                          learning_rate=0.0025, print_every=2000, save_every=1, image_path='./image/val2014_resized',
                          pretrained_model=None, model_path='./model/lstm', test_model=test_model+str(i),
                          print_bleu=True, log_path='./log/')

	# validation set
	#print "solver val score"
	#solver.test(data, split='val')
	#evaluate(data_path='./data', split='val')

	# test set
	print "solver test score of "
	test = load_coco_data(data_path=data_path, split=split)
	solver.test(test, split=split, save_path=data_path)

        final_scores = evaluate(data_path=data_path, split=split,get_scores=True)
	output += str(final_scores['Bleu_4'])+","+str(final_scores['METEOR'])+","+str(final_scores['CIDEr']) + ","
	output += str(final_scores['Bleu_4']+final_scores['METEOR']+final_scores['CIDEr']) + "\n"
	tf.get_variable_scope().reuse_variables()


#f=open(test_model[:-6]+dataset+'.'+split+'.scores.csv','wb')
#f.write(output)
#f.close()

print "Successfully write " + split + " scores to: " + test_model[:-6] + dataset + '.' + split + '.scores.csv'
