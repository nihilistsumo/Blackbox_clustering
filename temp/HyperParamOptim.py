import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from util.Evaluator import ClusterEvaluator
import optuna
import numpy as np
import random
import argparse
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
from clearml import Task
from sentence_transformers import models
from model.BBCluster import CustomSentenceTransformer, BBClusterLossModel
from experiments.treccar_run import prepare_cluster_data_train_only, prepare_cluster_data2
from experiments.ng20_run import prepare_cluster_data

def _run_fixed_lambda_bbcluster(train_batch_size, num_epochs, lambda_val, reg, use_model_device, warmup_frac=0.1,
                                model_name='distilbert-base-uncased', out_features=256):
    exp_task = Task.init(project_name='Optuna Hyperparam optim', task_name='trial')
    config_dict = {'lambda_val': lambda_val, 'reg': reg}
    config_dict = task.connect(config_dict)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available and using device: '+str(device))
    else:
        device = torch.device('cpu')
        print('CUDA not available, using device: '+str(device))
    word_embedding_model = models.Transformer(model_name)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    doc_dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=out_features,
                                   activation_function=nn.Tanh())

    model = CustomSentenceTransformer(modules=[word_embedding_model, pooling_model, doc_dense_model])
    loss_model = BBClusterLossModel(model=model, device=device,
                                        lambda_val=config_dict.get('lambda_val', lambda_val),
                                        reg_const=config_dict.get('reg', reg))

    train_dataloader = DataLoader(train_cluster_data, shuffle=True, batch_size=train_batch_size)
    evaluator = ClusterEvaluator.from_input_examples(val_cluster_data, use_model_device)

    warmup_steps = int(len(train_dataloader) * num_epochs * warmup_frac)  # 10% of train data

    model.to(device)

    # Train the model
    model.fit(train_objectives=[(train_dataloader, loss_model)],
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              logger=exp_task.get_logger())
    return evaluator(model)

def objective(trial):
    lambda_val = trial.suggest_float('lambda_val', LAMBDA_MIN, LAMBDA_MAX)
    reg = trial.suggest_float('reg', REG_MIN, REG_MAX)
    train_batch_size = trial.suggest_int('train_batch_size', 1, 2)
    return _run_fixed_lambda_bbcluster(train_batch_size, NUM_EPOCHS_PER_TRIAL, lambda_val, reg, use_model_device)

parser = argparse.ArgumentParser(description='Run hyperparameter optimization using Optuna')
parser.add_argument('-dt', '--dataset', default='trec')
parser.add_argument('-nd', '--num_doc')
parser.add_argument('-vs', '--val_samples', type=int, default=20)

### TRECCAR only ###
parser.add_argument('-in', '--input_dir', default='/home/sk1105/sumanta/trec_dataset')
parser.add_argument('-tin', '--train_input', default='train/base.train.cbor')
parser.add_argument('-tp', '--train_paratext', default='train/train_paratext/train_paratext.tsv')
parser.add_argument('-lv', '--level', default='top')
####################

parser.add_argument('-ep', '--num_epoch', type=int, default=2)
parser.add_argument('-lm', '--lambda_min', type=float, default=20.0)
parser.add_argument('-lx', '--lambda_max', type=float, default=200.0)
parser.add_argument('-rm', '--reg_min', type=float, default=0.0)
parser.add_argument('-rx', '--reg_max', type=float, default=10.0)
parser.add_argument('--gpu_eval', default=False, action='store_true')
args = parser.parse_args()

task = Task.init(project_name='Optuna Hyperparam optim', task_name='Optimization')
num_doc = args.num_doc
val_samples = args.val_samples
LAMBDA_MIN = args.lambda_min
LAMBDA_MAX = args.lambda_max
REG_MIN = args.reg_min
REG_MAX = args.reg_max
NUM_EPOCHS_PER_TRIAL = args.num_epoch
use_model_device = args.gpu_eval
if args.dataset == 'trec':
    input_dir = args.input_dir
    train_in = args.train_input
    train_pt = args.train_paratext
    train_art_qrels = input_dir + '/' + train_in + '-article.qrels'
    train_top_qrels = input_dir + '/' + train_in + '-toplevel.qrels'
    train_hier_qrels = input_dir + '/' + train_in + '-hierarchical.qrels'
    train_paratext = input_dir + '/' + train_pt
    val_art_qrels = input_dir + '/benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-article.qrels'
    val_top_qrels = input_dir + '/benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-toplevel.qrels'
    val_hier_qrels = input_dir + '/benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-hierarchical.qrels'
    val_paratext = input_dir + '/benchmarkY1/benchmarkY1-train-nodup/by1train_paratext/by1train_paratext.tsv'
    train_top_cluster_data, train_hier_cluster_data = prepare_cluster_data_train_only(train_art_qrels, train_top_qrels,
                                                                                      train_hier_qrels, train_paratext,
                                                                                      num_doc)
    print('Val data')
    val_top_cluster_data, val_hier_cluster_data = prepare_cluster_data2(val_art_qrels, val_top_qrels, val_hier_qrels,
                                                                        val_paratext, False, -1, val_samples)
    if args.level == 'top':
        train_cluster_data, val_cluster_data = train_top_cluster_data, val_top_cluster_data
    else:
        train_cluster_data, val_cluster_data = train_hier_cluster_data, val_hier_cluster_data
elif args.dataset == 'ng20':
    train_cluster_data, val_cluster_data, _ = prepare_cluster_data(num_doc, 1, val_samples)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)