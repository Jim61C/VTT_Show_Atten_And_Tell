
# coding: utf-8

# In[1]:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle as pickle
import tensorflow as tf
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
from core.bleu import evaluate

#get_ipython().magic(u'matplotlib inline')
#plt.rcParams['figure.figsize'] = (8.0, 6.0)  # set default size of plots
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')


# In[2]:

data = load_coco_data(data_path='./data', split='val')
with open('./data/train/word_to_idx.pkl', 'rb') as f:
    word_to_idx = pickle.load(f)


# In[3]:

model = CaptionGenerator(word_to_idx, dim_feature=[1, 2048], dim_embed=512,
                         dim_hidden=1024, n_time_step=16, prev2out=True,
                         ctx2out=False, alpha_c=0.0, selector=False, dropout=True)


# In[4]:

solver = CaptioningSolver(model, data, data, n_epochs=15, batch_size=128, update_rule='adam',
                          learning_rate=0.0025, print_every=2000, save_every=1, image_path='./image/val2014_resized',
                          pretrained_model=None, model_path='./model/lstm', test_model='./model/lstm/model-10',
                          print_bleu=True, log_path='./log/')


# In[7]:
print "solver val score"
solver.test(data, split='val')


# In[8]:
test = load_coco_data(data_path='./data', split='test')


# In[13]:
print "solver test score"
tf.get_variable_scope().reuse_variables()
solver.test(test, split='test')


# In[14]:

evaluate(data_path='./data', split='val')


# In[15]:

evaluate(data_path='./data', split='test')

