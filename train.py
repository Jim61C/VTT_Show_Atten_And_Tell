from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
import config
import sys
reload(sys)
sys.setdefaultencoding("utf-8") # to get rid of ASCII decoding issue


def main():
    # load train dataset
    data = load_coco_data(data_path=config.DATASET, split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path=config.DATASET, split='val')

    model = CaptionGenerator(word_to_idx, dim_feature=[64, 2048], dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, prev2out=True, 
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True, device_id = '/gpu:0')

    print "Model loaded, start to solve..."
    solver = CaptioningSolver(model, data, val_data, n_epochs=20, batch_size=98, update_rule='adam',
                                          learning_rate=0.001, print_every=1000, save_every=1, image_path='./image_{}/'.format(config.DATASET_SUFFIX),
                                    pretrained_model=None, model_path='model_{}/lstm/'.format(config.DATASET_SUFFIX), test_model='model/lstm/model-10',
                                     print_bleu=True, log_path='log_{}/'.format(config.DATASET_SUFFIX), data_path = config.DATASET )

    solver.train()

if __name__ == "__main__":
    main()