from core.solver import CaptioningSolver
from core.lstm_gan_model import CaptionGenerator
from core.utils import load_coco_data


def main():
    # load train dataset
    data = load_coco_data(data_path='./data_MSRVTT', split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path='./data_MSRVTT', split='val')

    model = CaptionGenerator(word_to_idx, dim_feature=[1, 2048], dim_embed=512,
                    dim_hidden=1024, n_time_step=16, prev2out=True,
                    ctx2out=False, alpha_c=0, selector=False, dropout=True)

    solver = CaptioningSolver(model, data, val_data, n_epochs=40, batch_size=256, update_rule='adam',
                    learning_rate=0.001, print_every=50, save_every=1, image_path='./image/',
                    pretrained_model='../show-attend-and-tell-MSRVTT_scratch/model/lstm_MSRVTT_from_scratch/model-6', model_path='model/show_and_tell/lstm_gan/', test_model='../show-attend-and-tell-MSRVTT_scratch/model/lstm_MSRVTT_from_scratch/model-6',
                    print_bleu=True, log_path='log/', reset_embedding=False, use_gan=True)

    solver.train_gan()

if __name__ == "__main__":
    main()
