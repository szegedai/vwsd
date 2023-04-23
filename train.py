import os, sys, itertools

from multilingual_clip import pt_multilingual_clip
from transformers import AutoTokenizer
import open_clip

import torch
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

from utils.sparser import FISTA

import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s')
logger = logging.getLogger(__name__)
#logger.setLevel(level=logging.DEBUG)


def process_text_batch(model, input_text, device, tokenizer):
    with torch.no_grad():
        if 'Multilingual' in str(type(model)): # M-CLIP performs tokenization differently to open_clip
            encoded_text = model(input_text, tokenizer)
        else:
            tokenized_text = tokenizer(input_text).to(device)
            encoded_text = model.encode_text(tokenized_text)
    return encoded_text

def get_token_representations(tokenizer_name, clip_model, device_name, input_dataset, batch_size=512):
    device = torch.device(device_name)

    tokenizer = None
    if tokenizer_name.lower().startswith('m-clip'):
        model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = open_clip.get_tokenizer(tokenizer_name)
        model, _, preproc = open_clip.create_model_and_transforms(clip_model[0], device=device, pretrained=clip_model[1])
        model.to(device)
    
    text_embeddings = []
    for batch_id in range(len(input_dataset) // batch_size):
        batch_text = [d[1] for d in input_dataset[batch_id*batch_size:(batch_id+1)*batch_size]]
        text_embeddings.append(process_text_batch(model, batch_text, device, tokenizer))
    if len(input_dataset[len(text_embeddings)*batch_size:]) > 0: # handling the last incomple batch
        batch_text = [d[1] for d in input_dataset[len(text_embeddings)*batch_size:]]
        text_embeddings.append(process_text_batch(model, batch_text, device, tokenizer))

    encoded_text = torch.Tensor(torch.vstack(text_embeddings))
    encoded_text /= encoded_text.norm(dim=-1, keepdim=True)
    return encoded_text.detach().to(device)

def process_data_point(data_point, im_vecs, im_alphas, img_emb_files, text_vec, text_alphas, basic_features_to_use, use_dense, use_sparse):
    # use_sparse = 0 => sparse features are not used
    # use_sparse = 1 => sparse features are used for the intersection coefficients only
    # use_sparse = 2 => sparse features are used for all coefficients
    embs, alphas = [], []
    for idx, im_file in enumerate(data_point):
        if im_file in img_emb_files:
            embs.append(im_vecs[img_emb_files[im_file]])
            alphas.append(im_alphas[img_emb_files[im_file]])
        else:
            embs.append(torch.zeros(im_vecs.shape[1], dtype=im_vecs.dtype).to(im_vecs.device))
            alphas.append(torch.zeros(im_alphas.shape[1], dtype=im_alphas.dtype).to(im_alphas.device))

    num_sparse_features = text_alphas.shape[0]
    txt_nnz_indices = set(torch.where(text_alphas!=0)[0].cpu().numpy())

    sparse_dots, dense_dots = [], [] # one value for each potential image for the given input
    feature_vectors = [] # one for each image for the given input

    for j in range(len(alphas)):
        im_nnz_indices = set(torch.where(alphas[j]!=0)[0].cpu().numpy())
        intersection = txt_nnz_indices & im_nnz_indices 
        union = txt_nnz_indices | im_nnz_indices

        elementwise_products = (embs[j] * text_vec).cpu().numpy() # elementwise product of the dense embeddings
        dense_dots.append(elementwise_products.sum())

        sparse_dots.append(0)
        sparse_features = np.zeros(num_sparse_features if use_sparse > 0 else 0)
        for base in union:
            in_intersection = base in im_nnz_indices and base in txt_nnz_indices
            if sparse_features.shape[0] > 0:
                sparse_features[base] = 1 if in_intersection else (-1 if use_sparse == 2 else 0)
            sparse_dots[-1] += (text_alphas[base] * alphas[j][base]).item()

        jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0
        default_features = []
        for t,f in zip(basic_features_to_use, [jaccard_similarity, sparse_dots[-1], dense_dots[-1]]):
            if t is True:
                default_features.append(f)

        feature_vector = elementwise_products if use_dense else np.zeros(0)
        feature_vectors.append(np.hstack([feature_vector, sparse_features, np.array(default_features)]))
    return feature_vectors, dense_dots, sparse_dots

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs predictions.')
    parser.add_argument('--clip_model', nargs='+', default=[#'ViT-B-32-quickgelu',
                                                            #'ViT-B-16-plus-240',
                                                            #'xlm-roberta-base-ViT-B-32',
                                                            'xlm-roberta-large-ViT-H-14',])
                                                            #'ViT-L_14',
                                                            #'ViT-B-16-plus-240'])
    parser.add_argument('--clip_pretrained', nargs='+', default=[#'laion400m_e32', 'laion400m_e32',
                                                                 #'laion5b_s13b_b90k', 
                                                                 'frozen_laion5b_s13b_b90k'])#, 'openai', 'laion400m_e32'])
    parser.add_argument('--tokenizer', nargs='+', default=[#'ViT-B-32-quickgelu',
                                                           #'ViT-B-16-plus-240',
                                                           #'xlm-roberta-base-ViT-B-32',
                                                           'xlm-roberta-large-ViT-H-14',])
                                                           #'M-CLIP/XLM-Roberta-Large-Vit-L-14',
                                                           #'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus'])
    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--K', type=int, nargs='+', default=[3000])
    parser.add_argument('--lda', type=float, nargs='+', default=[0.05])
    parser.add_argument('--lr_regularizer', type=float, default=1.0)

    parser.add_argument('--basic_features', type=int, default=7, help='stores the decimal equivalent of a 3 digit binary number indicating which basic features to use')
    parser.add_argument('--sparse_features', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--dense_features', dest='dense', action='store_true')
    parser.add_argument('--not_dense_features', dest='dense', action='store_false')
    parser.set_defaults(dense=True)

    parser.add_argument('--data_folder', default='/data/berend/vWSD_semeval23/')
    parser.add_argument('--data_file', default='train_v1/train.data.v1.txt')
    parser.add_argument('--gold_labels', default='train_v1/train.gold.v1.txt')
    parser.add_argument('--embeddings_path', default='train_v1/representations/')
    parser.add_argument('--training_instances', type=int, default=10000)

    parser.add_argument('--test_gold_file', nargs='+', default=['test/en.test.gold.v1.1.txt', 'test/it.test.gold.v1.1.txt', 'test/fa.test.gold.txt'])
    parser.add_argument('--test_data_file', nargs='+', default=['test/en.test.data.v1.1.txt', 'test/it.test.data.v1.1.txt', 'test/fa.test.data.txt'])
    parser.add_argument('--test_embeddings_path', default='test/representations/')

    args = parser.parse_args()
    logger.info(args)
    basic_features = [b=='1' for b in '{:03b}'.format(args.basic_features)]

    if torch.cuda.is_available() and torch.cuda.device_count() > args.gpu_id >= 0:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device("cpu")
    logging.info('Using {}'.format(device))

    golds = [l.strip() for l in open(args.data_folder + args.gold_labels)]
    dataset = [l.strip().split('\t') for l in open(args.data_folder + args.data_file)]

    Xs, params = [], []
    for cm, cp, tokenizer_name in zip(args.clip_model, args.clip_pretrained, args.tokenizer):

        image_embeddings_file = '{}/{}/{}-{}.npy'.format(args.data_folder, args.embeddings_path, cm, cp)
        im_vecs = torch.from_numpy(np.load(image_embeddings_file)).to(device)
        with open(image_embeddings_file.replace('.npy', '.pickle'), 'rb') as fo:
            img_emb_files = {file_name:i for i,file_name in enumerate(pickle.load(fo))}
        txt_vecs = get_token_representations(tokenizer_name, (cm, cp), device, dataset)

        for l in args.lda:
            for K in args.K:
                params.append([cm, cp, tokenizer_name, l, K])
    
                D = torch.from_numpy(np.load('{}/{}/{}-{}_{}_{}.npy'.format(args.data_folder, args.embeddings_path, cm, cp, K, l))).to(device)
                im_vecs = torch.from_numpy(np.load(image_embeddings_file)).to(device)
                im_alphas = FISTA(im_vecs.T, D, l, 100)[0].T
                txt_alphas = FISTA(txt_vecs.T, D, l, 100)[0].T

                logger.info((cm, cp, l, K))
                X_train, y_train = [], []
                corrects = {k:[] for k in range(2)}
                for i,d in enumerate(dataset):
                    if i==args.training_instances: break
                    if i>0 and i%2500==0: logger.info("{} / {}".format(i, len(dataset)))
                    #(data_point, im_vecs, im_alphas, img_emb_files, text_alphas, use_dense, use_sparse)
                    features, dense_dots, sparse_dots = process_data_point(d[2:], im_vecs, im_alphas, img_emb_files, txt_vecs[i], txt_alphas[i], basic_features, args.dense, args.sparse_features)

                    X_train.extend(features)

                    for idx, im_file in enumerate(d[2:]):
                        y_train.append(im_file==golds[i])

                        if im_file==golds[i]: gold_index = idx

                    #for k, dots in enumerate([dense_dots, sparse_dots]):
                    #    corrects[k].append(len(dots) - np.where(np.argsort(dots)==gold_index)[0][0])

                Xs.append(X_train)
                #for k,v in corrects.items():
                #    for from_value in [0, args.training_instances]:
                #        acc = sum([1 for rank in v[from_value:] if rank==1]) / len(v[from_value:])
                #        mrr = np.mean([1/rank for rank in v[from_value:]])
                #        logger.info('{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}'.format(k, from_value, params[-1], len(v[from_value:]), mrr, acc))

    logger.info('Num features: {}'.format(len(Xs[0][0])))
    models = []
    probas = np.zeros((10, len(X_train)//10, len(Xs)))
    for xi, X in enumerate(Xs):
        gold_indices = []
        l = LogisticRegression(random_state=0, max_iter=5000, C=args.lr_regularizer, penalty='l2', solver='lbfgs', fit_intercept=True).fit(X, y_train)
        models.append(l)
        '''
        test_ranks = []
        for j in range((len(X)-threshold)//10):
            index_from, index_to = threshold+j*10, threshold+(j+1)*10
            p = l.predict_proba(X[index_from:index_to])
            probas[:,j, ci*len(Xs)+xi] += p[:,1]
            gold_index = np.where(y[index_from:index_to])[0][0]
            gold_indices.append(gold_index)
            rank = 10 - np.where(np.argsort(p[:,1])==gold_index)[0][0]
            test_ranks.append(rank)
        acc = sum([tr==1 for tr in test_ranks]) / len(test_ranks)
        mrr = np.mean([1/tr for tr in test_ranks])
        logger.info("{}\t{}\tC={:.2f}\t{:.4f}\t{:.4f}".format(X[0].shape, params[xi], c, mrr, acc))
       '''

    test_fts, y_test = [], []
    if len(dataset) > args.training_instances:
        test_probas = np.zeros((10, len(dataset) - args.training_instances, len(models)))
        logger.info("test_probas {}".format(test_probas.shape))

        for model_num, (model, parameters) in enumerate(zip(models, params)):
            cm, cp, tokenizer_name, l, K = parameters
            test_image_embeddings_file = '{}/{}/{}-{}.npy'.format(args.data_folder, args.embeddings_path, cm, cp)
            test_im_vecs = torch.from_numpy(np.load(test_image_embeddings_file)).to(device)

            with open(test_image_embeddings_file.replace('.npy', '.pickle'), 'rb') as fo:
                test_img_emb_files = {file_name:i for i,file_name in enumerate(pickle.load(fo))}
            test_txt_vecs = get_token_representations(tokenizer_name, (cm, cp), device, dataset)

            D = torch.from_numpy(np.load('{}/{}/{}-{}_{}_{}.npy'.format(args.data_folder, args.embeddings_path, cm, cp, K, l))).to(device)
            test_im_alphas = FISTA(test_im_vecs.T, D, l, 100)[0].T
            test_txt_alphas = FISTA(test_txt_vecs.T, D, l, 100)[0].T

            ranks = {'dense': [], 'sparse':[], 'model':[]}
            for i,d in enumerate(dataset):
                if i < args.training_instances: continue
                test_features, test_dense_dots, test_sparse_dots = process_data_point(d[2:], test_im_vecs, test_im_alphas, test_img_emb_files, test_txt_vecs[i], test_txt_alphas[i], basic_features, args.dense, args.sparse_features)
                test_fts.extend(test_features)
                for idx, im_file in enumerate(d[2:]):
                    y_test.append(im_file==golds[i])
                probs = model.predict_proba(test_features)[:,1]
                test_probas[:, i-args.training_instances, model_num] = probs
                scores_to_rank = lambda scores: 1 + [d[2:][r] for r in np.argsort(scores)[::-1]].index(golds[i]) 
                ranks['dense'].append(scores_to_rank(test_dense_dots))
                ranks['sparse'].append(scores_to_rank(test_sparse_dots))
                ranks['model'].append(scores_to_rank(probs))
            logger.info((cm, cp, l, K))
            for k,v in ranks.items():
                mrr = np.mean([1/r for r in v])
                acc = np.mean([r==1 for r in v])
                logger.info('{}\t{}\t{:.4f}\t{:.4f}\t{}'.format('train', k if k!='model' else '_'.join(map(str, parameters)), mrr, acc, len(v)))


    for tdf, tgf in zip(args.test_data_file, args.test_gold_file):
        test_fts2, y_test2 = [], []
        test_dataset2 = [l.strip().split('\t') for l in open(args.data_folder + tdf)]
        test_golds2 = [l.strip() for l in open(args.data_folder + tgf)]
        test_probas2 = np.zeros((10, len(test_dataset2), len(models)))
        for model_num, (model, parameters) in enumerate(zip(models, params)):
            cm, cp, tokenizer_name, l, K = parameters
            if l==0.1 and K==3000 and tokenizer_name=='xlm-roberta-large-ViT-H-14':
                pickle.dump(model, open('{}_{}_{}_{}.pickle'.format(tokenizer_name, K, l, args.sparse_features), 'wb'))
            test_image_embeddings_file2 = '{}/{}/{}-{}.npy'.format(args.data_folder, args.test_embeddings_path, cm, cp)
            test_im_vecs2 = torch.from_numpy(np.load(test_image_embeddings_file2)).to(device)

            with open(test_image_embeddings_file2.replace('.npy', '.pickle'), 'rb') as fo:
                test_img_emb_files2 = {file_name:i for i,file_name in enumerate(pickle.load(fo))}
            test_txt_vecs2 = get_token_representations(tokenizer_name, (cm, cp), device, test_dataset2)

            D = torch.from_numpy(np.load('{}/{}/{}-{}_{}_{}.npy'.format(args.data_folder, args.embeddings_path, cm, cp, K, l))).to(device)
            test_im_alphas2 = FISTA(test_im_vecs2.T, D, l, 100)[0].T
            test_txt_alphas2 = FISTA(test_txt_vecs2.T, D, l, 100)[0].T

            ranks = {'dense': [], 'sparse':[], 'model':[]}
            for i,d in enumerate(test_dataset2):
                test_features, test_dense_dots, test_sparse_dots = process_data_point(d[2:], test_im_vecs2, test_im_alphas2, test_img_emb_files2, test_txt_vecs2[i], test_txt_alphas2[i], basic_features, args.dense, args.sparse_features)
                test_fts2.extend(test_features)
                for idx, im_file in enumerate(d[2:]):
                    y_test2.append(im_file==test_golds2[i])
                probs = model.predict_proba(test_features)[:,1]
                test_probas[:, i, model_num] = probs
                scores_to_rank = lambda scores: 1 + [d[2:][r] for r in np.argsort(scores)[::-1]].index(test_golds2[i]) 
                ranks['dense'].append(scores_to_rank(test_dense_dots))
                ranks['sparse'].append(scores_to_rank(test_sparse_dots))
                ranks['model'].append(scores_to_rank(probs))
            logger.info((cm, cp, l, K))
            for k,v in ranks.items():
                mrr = np.mean([1/r for r in v])
                acc = np.mean([r==1 for r in v])
                logger.info('{}\t{}\t{:.4f}\t{:.4f}\t{}'.format(tdf, k if k!='model' else '_'.join(map(str, parameters)), mrr, acc, len(v)))

    sys.exit(2)
    best_val = 0
    for e, selections in enumerate(itertools.product([True, False], repeat=len(Xs))):
        if sum(selections)==0: continue
        aggregated_probs = np.sum(probas[:,:,selections], -1)
        ensemble_ranks = [10-np.where(np.argsort(aggregated_probs[:,k], axis=0)==gi)[0][0] for k, gi in enumerate(gold_indices)]
        acc = sum([tr==1 for tr in ensemble_ranks]) / len(test_ranks)
        mrr = np.mean([1/tr for tr in ensemble_ranks])
        if best_val < acc:
            best_val = acc
            logger.info("{}\t{}\t{:.4f}\t{:.4f}\t{}".format(e, sum(selections), mrr, acc, selections))

    for tdf in args.test_data_file:
        test_dataset = [l.strip().split('\t') for l in open(args.data_folder + tdf)]
        test_probas = np.zeros((10, len(test_dataset), len(models)))
        for model_num, (model, parameters) in enumerate(zip(models, params)):
            cm, cp, tokenizer_name, l, K = parameters
            test_image_embeddings_file = '{}/{}/{}-{}.npy'.format(args.data_folder, args.test_embeddings_path, cm, cp)
            test_im_vecs = torch.from_numpy(np.load(test_image_embeddings_file)).to(device)

            with open(test_image_embeddings_file.replace('.npy', '.pickle'), 'rb') as fo:
                test_img_emb_files = {file_name:i for i,file_name in enumerate(pickle.load(fo))}
            test_txt_vecs = get_token_representations(tokenizer_name, (cm, cp), device, test_dataset)

            D = torch.from_numpy(np.load('{}/{}/{}-{}_{}_{}.npy'.format(args.data_folder, args.embeddings_path, cm, cp, K, l))).to(device)
            test_im_alphas = FISTA(test_im_vecs.T, D, l, 100)[0].T
            test_txt_alphas = FISTA(test_txt_vecs.T, D, l, 100)[0].T

            logger.info((cm, cp, l, K))
            for i,d in enumerate(test_dataset):
                test_features, test_dense_dots, test_sparse_dots = process_data_point(d[2:], test_im_vecs, test_im_alphas, test_img_emb_files, test_txt_vecs[i], test_txt_alphas[i], basic_features, args.dense, args.sparse_features)
                probs = model.predict_proba(test_features)[:,1]
                test_probas[:, i, model_num] = probs

        for model_num, model_params in enumerate(params):
            sorted_indices = np.argsort(-test_probas[:,:,model_num], axis=0)
            with open('{}{}_{}_predictions.txt'.format(args.data_folder, tdf, model_num), 'w') as fo:
                for test_instance_id in range(sorted_indices.shape[1]):
                    potential_filenames = test_dataset[test_instance_id][2:]
                    fo.write('\t'.join([potential_filenames[si] for si in sorted_indices[:, test_instance_id]])+'\n')

        aggregated_probas = test_probas.sum(axis=-1) # one column per test instances
        sorted_indices = np.argsort(-aggregated_probas, axis=0)
        with open(args.data_folder + tdf + '_merged_predictions.txt', 'w') as fo:
            for test_instance_id in range(sorted_indices.shape[1]):
                potential_filenames = test_dataset[test_instance_id][2:]
                fo.write('\t'.join([potential_filenames[si] for si in sorted_indices[:, test_instance_id]])+'\n')
