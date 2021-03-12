from experiments.treccar_run import get_trec_dat, get_paratext_dict
from util.Data import InputTRECCARExample
from util.Evaluator import ClusterEvaluator
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.readers import InputExample
from tqdm.autonotebook import trange
import argparse
from sklearn.datasets import fetch_20newsgroups
import torch

def evaluate_treccar(model_path, test_art_qrels, test_top_qrels, test_hier_qrels, test_paratext, level):
    test_page_paras, test_rev_para_top, test_rev_para_hier = get_trec_dat(test_art_qrels, test_top_qrels,
                                                                          test_hier_qrels)
    test_len_paras = np.array([len(test_page_paras[page]) for page in test_page_paras.keys()])
    print('test mean paras: %.2f, std: %.2f, max paras: %.2f' % (np.mean(test_len_paras), np.std(test_len_paras),
                                                                 np.max(test_len_paras)))
    test_ptext_dict = get_paratext_dict(test_paratext)
    test_top_cluster_data = []
    test_hier_cluster_data = []
    max_num_doc_test = max([len(test_page_paras[p]) for p in test_page_paras.keys()])
    test_pages = list(test_page_paras.keys())
    for i in trange(len(test_pages)):
        page = test_pages[i]
        paras = test_page_paras[page]
        paratexts = [test_ptext_dict[p] for p in paras]
        top_sections = list(set([test_rev_para_top[p] for p in paras]))
        top_labels = [top_sections.index(test_rev_para_top[p]) for p in paras]
        hier_sections = list(set([test_rev_para_hier[p] for p in paras]))
        hier_labels = [hier_sections.index(test_rev_para_hier[p]) for p in paras]
        query_text = ' '.join(page.split('enwiki:')[1].split('%20'))
        n = len(paras)
        paras = paras + ['dummy'] * (max_num_doc_test - n)
        paratexts = paratexts + [''] * (max_num_doc_test - n)
        top_labels = top_labels + [-1] * (max_num_doc_test - n)
        hier_labels = hier_labels + [-1] * (max_num_doc_test - n)
        test_top_cluster_data.append(InputTRECCARExample(qid=page, q_context=query_text, pids=paras, texts=paratexts,
                                                         label=np.array(top_labels)))
        test_hier_cluster_data.append(InputTRECCARExample(qid=page, q_context=query_text, pids=paras, texts=paratexts,
                                                          label=np.array(hier_labels)))
    print("Top-level datasets")
    print("Test instances: %5d" % len(test_top_cluster_data))

    model = SentenceTransformer(model_path)
    if level == 'h':
        print('Evaluating hiererchical clusters')
        test_evaluator = ClusterEvaluator.from_input_examples(test_hier_cluster_data)
        model.evaluate(test_evaluator)
    else:
        print('Evaluating toplevel clusters')
        test_evaluator = ClusterEvaluator.from_input_examples(test_top_cluster_data)
        model.evaluate(test_evaluator)

def evaluate_ng20(model_path, test_cluster_data):
    if torch.cuda.is_available():
        print('CUDA is available')
        device = torch.device('cuda')
    else:
        print('Using CPU')
        device = torch.device('cpu')
    model = SentenceTransformer(model_path)
    model.to(device)
    test_evaluator = ClusterEvaluator.from_input_examples(test_cluster_data)
    model.evaluate(test_evaluator)

def main():
    parser = argparse.ArgumentParser(description='Evaluate saved models')
    parser.add_argument('-dt', '--data', default='trec')
    parser.add_argument('-in', '--input_dir', default='/home/sk1105/sumanta/trec_dataset')
    parser.add_argument('-mp', '--model_path')
    parser.add_argument('-lv', '--level', default='t')

    args = parser.parse_args()
    dataset = args.data
    input_dir = args.input_dir
    model_path = args.model_path
    level = args.level
    if dataset == 'trec':
        test_art_qrels = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-article.qrels'
        test_top_qrels = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-toplevel.qrels'
        test_hier_qrels = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-hierarchical.qrels'
        test_paratext = input_dir + '/benchmarkY1/benchmarkY1-test-nodup/by1test_paratext/by1test_paratext.tsv'

        evaluate_treccar(model_path, test_art_qrels, test_top_qrels, test_hier_qrels, test_paratext, level)
    elif dataset == '20ng':
        pages_to_cluster = 50
        ng_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
        test_cluster_data = []
        for i in range(len(ng_test['filenames']) // pages_to_cluster):
            test_cluster_data.append(
                InputExample(texts=ng_test['data'][i * pages_to_cluster: (i + 1) * pages_to_cluster],
                             label=ng_test['target'][
                                   i * pages_to_cluster: (i + 1) * pages_to_cluster]))
        print("Test instances: %5d" % len(test_cluster_data))
        evaluate_ng20(model_path, test_cluster_data)

if __name__ == '__main__':
    main()
