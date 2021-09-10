from sentence_transformers import models
from sentence_transformers.util import batch_to_device
import torch
import torch.nn as nn
from model.BBCluster import CustomSentenceTransformer
import sys

def do_test(pt_file, model_name, n):
    text = []
    i = 0
    with open(pt_file, 'r', encoding='utf8') as f:
        for l in f:
            text.append(l.split('\t')[1])
            i+=1
            if i >= n:
                break
    psg_word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    psg_pooling_model = models.Pooling(psg_word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True,
                                       pooling_mode_cls_token=False,
                                       pooling_mode_max_tokens=False)

    psg_dense_model = models.Dense(in_features=psg_pooling_model.get_sentence_embedding_dimension(),
                                   out_features=256,
                                   activation_function=nn.Tanh())
    psg_model = CustomSentenceTransformer(modules=[psg_word_embedding_model, psg_pooling_model, psg_dense_model])
    if torch.cuda.is_available():
        psg_model.to(torch.device('cuda'))
    psg_features = []
    print('Tokenizing')
    for p in text:
        psg_tkn = psg_model.tokenize(p)
        if torch.cuda.is_available():
            batch_to_device(psg_tkn, torch.device('cuda'))
        psg_features.append(psg_tkn)
    psg_embs = []
    print('Embedding')
    for pfet in psg_features:
        psg_emb = psg_model(pfet)['sentence_embedding']
        psg_emb.to(torch.device('cpu'))
        psg_embs.append(psg_emb)
    print(psg_embs[:10])

do_test(sys.argv[1], sys.argv[2], int(sys.argv[3]))
