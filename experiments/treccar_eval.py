import argparse
import pickle
import torch
from sentence_transformers import SentenceTransformer
from util.Evaluator import ClusterEvaluator
from experiments.treccar_run import prepare_cluster_data2

parser = argparse.ArgumentParser(description='Eval treccar experiments')
parser.add_argument('-ip', '--input_dir')
parser.add_argument('-lv', '--level', default='top')
parser.add_argument('-mp', '--model_paths', nargs='+')
parser.add_argument('--gpu_eval', default=False, action='store_true')
args = parser.parse_args()
input_dir = args.input_dir
level = args.level
model_paths = args.model_paths
gpu_eval = args.gpu_eval

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available and using device: ' + str(device))
else:
    device = torch.device('cpu')
    print('CUDA not available, using device: ' + str(device))

test_art_qrels = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-article.qrels'
test_top_qrels = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-toplevel.qrels'
test_hier_qrels = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-hierarchical.qrels'
test_paratext = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/by1test_paratext/by1test_paratext.tsv'
test_top_cluster_data, test_hier_cluster_data = prepare_cluster_data2(test_art_qrels, test_top_qrels, test_hier_qrels,
                                                                        test_paratext, False, -1, 0)
if level == 'top':
    test_cluster_data = test_top_cluster_data
else:
    test_cluster_data = test_hier_cluster_data
test_evaluator = ClusterEvaluator.from_input_examples(test_cluster_data, gpu_eval)
for mp in model_paths:
    m = SentenceTransformer(mp)
    m.to(device)
    print('Model: '+mp.split('/')[len(mp.split('/'))-1])
    m.evaluate(test_evaluator)