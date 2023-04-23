import os

import torch
import pickle
import numpy as np

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils.sparser import FISTA

import open_clip
import spams

import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

class Preproc(object):

    def __init__(self, clip_model, device):
        self.device = device
        self.model, _, self.preproc = open_clip.create_model_and_transforms(clip_model[0], device=device, pretrained=clip_model[1])

    def obtain_images(self, paths):
        images, ok_images = [], []
        for fn in paths:
            try:
                im = Image.open(fn).convert('RGB')
                images.append(self.preproc(im).unsqueeze(0))
                ok_images.append(os.path.basename(fn))
            except Exception as e:
               logger.info((e, fn))
        return images, ok_images

    def process_images(self, images, normalize=True):
        encoded_images = None
        with torch.no_grad():
            encoded_images = self.model.encode_image(torch.vstack(images).to(self.device))
            if normalize:
                encoded_images /= encoded_images.norm(dim=-1, keepdim=True)
        return encoded_images.detach().cpu().numpy()

    '''
    def process_text(self, texts, normalize=True):
        encoded_text = None
        with torch.no_grad():
            if self.mclip:
                encoded_text = self.multiling_model(texts, self.tokenizer).to(self.device)
            else:
                text_inputs = open_clip.tokenize(texts).to(self.device)
                encoded_text = self.model.encode_text(text_inputs)
            if normalize:
                encoded_text /= encoded_text.norm(dim=-1, keepdim=True)
        return encoded_text.detach().cpu().numpy()
    '''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Determines CLIP representations and performes dictionary learning.')
    #parser.add_argument('--clip_model', nargs='+', default=('ViT-B-32-quickgelu', 'laion400m_e32'))#, required=True)
    #parser.add_argument('--clip_model', nargs='+', default=('ViT-B-16-plus-240', 'laion400m_e32'))
    #parser.add_argument('--clip_model', nargs='+', default=('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'))
    #parser.add_argument('--clip_model', nargs='+', default=('ViT-L/14', 'openai')) # used by https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-L-14
    #parser.add_argument('--clip_model', nargs='+', default=('ViT-B-16-plus-240', 'laion400m_e32')) # used by https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-B-16Plus
    parser.add_argument('--clip_model', nargs='+', default=('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k'))
    
    parser.add_argument('--gpu_id', type=int, default=0, help='default:0')
    parser.add_argument('--batch_size', type=int, default=256, help='default:256')
    parser.add_argument('--K', type=int, default=1000, help='default:0')
    parser.add_argument('--lda', type=float, default=0.05, help='default:0.05')
    parser.add_argument('--images_dir', default='/data/berend/vWSD_semeval23/trial_v1/trial_images_v1/')
    parser.add_argument('--out_dir', default='./representations/')

    args = parser.parse_args()

    if torch.cuda.is_available() and torch.cuda.device_count() > args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device("cpu")

    dirname = os.path.dirname(args.out_dir)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
        logger.info('{} created'.format(dirname))

    logger.info(args)

    embeddings_file = '{}/{}.npy'.format(dirname, '-'.join(args.clip_model).replace('/', '_'))

    p = Preproc(args.clip_model, device)

    if args.images_dir is not None and os.path.exists(embeddings_file):
        embeddings = np.load(embeddings_file)
    elif args.images_dir is not None:
  
        file_names = sorted([f for f in os.listdir(args.images_dir)])

        vecs, ok_images = [], []
        for bi in range(0, len(file_names), args.batch_size):
            logger.info((bi, len(ok_images)))
            batch_files = ['{}/{}'.format(args.images_dir, fn) for fn in file_names[bi:bi+args.batch_size]]
            images, ok_files = p.obtain_images(batch_files)
            ok_images.extend(ok_files)
            vecs.append(p.process_images(images))

        embeddings = np.vstack(vecs).astype('float32')
        np.save(embeddings_file, embeddings)
        with open(embeddings_file.replace('.npy', '.pickle'), 'wb') as fo:
            pickle.dump(ok_images, fo)

    if args.images_dir is not None and args.K > 0:
        params = {'K': args.K, 'lambda1': args.lda, 'numThreads': 8, 'iter': 1000, 'batchsize': 400, 'posAlpha': True, 'verbose': False}
        D = spams.trainDL(embeddings.T, **params)
        np.save('{}/{}_{}_{}'.format(dirname, '-'.join(args.clip_model).replace('/', '_'), args.K, args.lda),  D)
        logger.info((embeddings.shape, D.shape))
