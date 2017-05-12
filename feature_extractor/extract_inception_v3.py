import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import argparse
import cPickle
import tarfile
import json
import urllib
import sys
import time
import math
from multiprocessing import Pool, cpu_count
import skvideo.io
import skimage.transform
from skimage import exposure

# Functions and classes for loading and using the Inception model.
MULTI_WORKER = True

def create_dir(path):
  if (not os.path.exists(path)):
    os.makedirs(path)
    return path

class FeatureExtractor(object):
  def __init__(self, feature_tensor_name, model_dir = 'tmp/imagenet', \
    DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'):
    self.feature_tensor_name = feature_tensor_name
    self.model_dir = model_dir
    self.DATA_URL = DATA_URL

    self.maybe_download_and_extract()
    self.create_graph()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)

  def create_graph(self):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
        self.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name='')

  def extract_feature_one_image(self, img):
    """
    Input: img -- opencv read in images
    Returns spatial pooling features on this image, (w, h, c)
    """

    #Format for the Mul:0 Tensor
    np_image_data = skimage.transform.resize(img, (299, 299))
    #Numpy array, skimage is already RGB, no need inversion
    np_image_data = exposure.rescale_intensity(np_image_data.astype('float'), out_range=(-0.5, 0.5))
    np_image_data = np.expand_dims(np_image_data, axis=0)

    #now feeding it into the session:
    #[... initialization of session and loading of graph etc]

    feature_tensor = self.sess.graph.get_tensor_by_name(self.feature_tensor_name)
    features = self.sess.run(feature_tensor,
                               {'Mul:0': np_image_data})

    # print "features before squeeze:", features.shape
    # features = np.squeeze(features)
    # print "features after squeeze:", features.shape
    return features


  def extract_feature_video(self, video_file, start_sec, end_sec, final_feature_size):
    """
    Input:
    video_file -- the path to the clip
    start_sec -- start seconds to extract frame feature
    end_sec -- end seconds (exclusive)
    final_feature_size -- tuple of expected return feature size (after average)
    """
    # Creates graph from saved GraphDef.

    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.

    # temp_writer = tf.summary.FileWriter('temp_logs/',
    #                                   sess.graph)
    reader = skvideo.io.FFmpegReader(str(video_file))
    start_frame = 0
    frame_count = reader.getShape()[0]
    end_frame = frame_count -1

    print start_frame, "...", end_frame
    print "video_file: ", video_file


    total_feature_size = list(final_feature_size)
    total_feature_size.insert(0, 0)

    total_feature = np.empty((tuple(total_feature_size)))
    # print "total_feature.shape:",total_feature.shape

    for cur_frame in range(start_frame, end_frame + 1):
      try:
        frame = reader.nextFrame().next()
        # print "self.extract_feature_one_image for ", cur_frame
        # start_time = time.time()
        this_feature = self.extract_feature_one_image(frame)
        # print "time spent for extract_feature_one_image for frame:", cur_frame, " = ", time.time() - start_time
        total_feature = np.concatenate((total_feature, this_feature), axis = 0)

        # log
        if ((cur_frame - start_frame) % 10 == 0):
          print "extract feature {cur_frame}/{end_frame}, {video_file}".format(cur_frame = cur_frame, \
            end_frame = min(end_frame + 1, frame_count), video_file = video_file)
      except:
        print "frame ", cur_frame, " no feature extracted"

    print "total_feature.shape: ", total_feature.shape
    return np.mean(total_feature, axis = 0)

  def maybe_download_and_extract(self):
    """Download and extract model tar file."""
    dest_directory = self.model_dir
    if not os.path.exists(dest_directory):
      os.makedirs(dest_directory)
    filename = self.DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.urlretrieve(self.DATA_URL, filepath, _progress)
      print()
      statinfo = os.stat(filepath)
      print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def worker(args):
    clips, videos_folder, video_path_template, save_folder = args
    extactor = FeatureExtractor('mixed_10/join:0')
    avg_mided_10_feature_shape = (8,8,2048)

    for i in range(0, len(clips)):
    # for i in range(0, 5):
      this_clip_id = clips[i][:clips[i].find('.')]

      # if feature already extracted, skip
      if (os.path.exists(save_folder + str(this_clip_id) + '/' + 'mixed_10_join_feature.pkl')):
        print "Feature already extracted for video ", this_clip_id, ", {i}/{n}".format(i = i, \
          n = len(clips))
        continue

      this_video_path = video_path_template.format(this_clip_id)

      this_feature = None
      if (os.path.exists(this_video_path)):
        mixed_10_join_feature = extactor.extract_feature_video(this_video_path, \
          None, None, avg_mided_10_feature_shape)
        if (not mixed_10_join_feature is None):
          this_feature = mixed_10_join_feature.reshape(-1, 2048)

      if (not this_feature is None):
        print "Finished extracting mixed_10_join_feature for video", this_clip_id, ", {i}/{n}".format(i = i, \
          n = len(clips))
      elif(not os.path.exists(this_video_path)):
        print "Corresponding video of {video_id} not available for clip ".format(\
          video_id = this_clip_id), this_clip_id, ", {i}/{n}".format(i = i, \
          n = len(clips))
      else:
        print "Video {video_path} not valid, no feature extracted for ".format(\
          video_path = this_video_path), this_clip_id, ", {i}/{n}".format(i = i, \
          n = len(clips))


      # dump this feature
      create_dir(save_folder + str(this_clip_id) + '/')
      f = open(save_folder + str(this_clip_id) + '/' + 'mixed_10_join_feature.pkl', 'wb')
      cPickle.dump(this_feature, f, protocol = cPickle.HIGHEST_PROTOCOL)
      f.close()


def main():
    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #   '--model_dir',
    #   type=str,
    #   default='tmp/imagenet',
    #   help="""\
    #   Path to classify_image_graph_def.pb,
    #   imagenet_synset_to_human_label_map.txt, and
    #   imagenet_2012_challenge_label_map_proto.pbtxt.\
    #   """
    # )

    # parser.add_argument(
    #     '--image_file',
    #     type=str,
    #     default='tmp/imagenet/cropped_panda.jpg',
    #     help='Absolute path to image file.'
    # )

    # global FLAGS
    # FLAGS, unparsed = parser.parse_known_args()

    # print "FLAGS parsed: ", FLAGS.image_file
    videos_folder = 'YouTubeClips/'
    video_path_template = videos_folder + '{}.avi'
    save_folder = 'mixed_10_join_feature_dict/'
    create_dir(save_folder)
    clips = []
    for file_name in os.listdir(videos_folder):
      if (file_name.find('.avi') != -1):
        clips.append(file_name)

    ################## create splits ##################
    # split_file_template = "splits/split{}.txt"
    # num_split = 6 # since 8 GPU
    # worker_share = (int)(math.ceil(len(clips)/float(num_split)))
    # for i in range(0, num_split):
    #   print "split: ", i*worker_share, " to ", min((i+1)*worker_share, len(clips))
    #   split_indexes = range(i*worker_share, min((i+1)*worker_share, len(clips)))
    #   with open(split_file_template.format(i), 'wb') as f:
    #     f.write('\n'.join([str(idx) for idx in split_indexes]))

    # raise ValueError('purpose stop')

    ################## start feature extraction ##################
    # split_file = sys.argv[1]
    # print "split_file:", split_file
    # this_split = np.genfromtxt(split_file, dtype= int)
    # this_clips = []
    # for idx in this_split:
    #   this_clips.append(clips[idx])

    list_file = sys.argv[1]
    this_split = [0] # dummy
    this_clips = []
    for line in open(list_file, 'rb').readlines():
      line = line.replace('\n','')
      this_clips.append(line + '.mp4')

    if MULTI_WORKER:
      # each split use 3 workers
      num_workers = 3
      args = []

      worker_share = (int)(math.ceil(len(this_clips)/float(num_workers)))
      for i in range(0, num_workers):
        print "split: ", np.min(this_split) + i*worker_share, " to ", np.min(this_split) + min((i+1)*worker_share, len(this_clips))
        this_worker_clips = this_clips[i*worker_share:min((i+1)*worker_share, len(this_clips))]
        args.append((this_worker_clips, videos_folder, video_path_template, save_folder))

      pool = Pool(num_workers)
      pool.map(worker, args)
      pool.close()
      pool.join()
      print "Finished Parallel Feature Extraction!"

    else:
      worker((this_clips, videos_folder, video_path_template, save_folder))


if __name__ == "__main__":
    if len(sys.argv) != 2:
      print "Usage: python {} split_file".format(sys.argv[0])
      print "split_file -- E.g., splits/split0.txt"
      exit()
    main()
