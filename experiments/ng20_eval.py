import argparse
import pickle
import torch
from sentence_transformers import SentenceTransformer
from util.Evaluator import ClusterEvaluator

parser = argparse.ArgumentParser(description='Eval 20 news groups experiments')
parser.add_argument('-tp', '--test_data')
parser.add_argument('-mp', '--model_paths', nargs='+')
args = parser.parse_args()
test_data_path = args.test_data
model_paths = args.model_paths

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available and using device: ' + str(device))
else:
    device = torch.device('cpu')
    print('CUDA not available, using device: ' + str(device))

with open(test_data_path, 'rb') as f:
    test_cluster_data = pickle.load(f)
test_evaluator = ClusterEvaluator.from_input_examples(test_cluster_data)
for mp in model_paths:
    m = SentenceTransformer(mp)
    m.to(device)
    print('Model: '+mp.split('/')[len(mp.split('/'))-1])
    m.evaluate(test_evaluator)