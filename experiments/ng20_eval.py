import argparse
import pickle
from sentence_transformers import SentenceTransformer
from util.Evaluator import ClusterEvaluator

parser = argparse.ArgumentParser(description='Eval 20 news groups experiments')
parser.add_argument('-tp', '--test_data')
parser.add_argument('-mp', '--model_paths', nargs='+')
args = parser.parse_args()
test_data_path = args.test_data
model_paths = args.model_paths

with open(test_data_path, 'rb') as f:
    test_cluster_data = pickle.load(f)
test_evaluator = ClusterEvaluator.from_input_examples(test_cluster_data)
for mp in model_paths:
    m = SentenceTransformer(mp)
    print('Model: '+mp.split('/')[len(mp.split('/'))-1])
    m.evaluate(test_evaluator)