from experiments.treccar_run import get_trec_dat, get_paratext_dict
from collections import Counter
import numpy as np
import sys
import json

def trec_stats(art_qrels, top_qrels, hier_qrels, paratext_file, i):
    page_paras, rev_para_top, rev_para_hier = get_trec_dat(art_qrels, top_qrels, hier_qrels)
    paratext_dict = get_paratext_dict(paratext_file)
    arts = []
    stats = []
    c = 0
    for p in page_paras.keys():
        paras = page_paras[p]
        n = len(paras)
        top_section_counts = list(Counter([rev_para_top[p] for p in paras]).values())
        hier_section_counts = list(Counter([rev_para_hier[p] for p in paras]).values())
        top_k = len(top_section_counts)
        mean_top_k = np.mean(top_section_counts)
        std_top_k = np.std(top_section_counts)
        min_top_k = min(top_section_counts)
        max_top_k = max(top_section_counts)
        hier_k = len(hier_section_counts)
        mean_hier_k = np.mean(hier_section_counts)
        std_hier_k = np.std(hier_section_counts)
        min_hier_k = min(hier_section_counts)
        max_hier_k = max(hier_section_counts)
        lens = [len(paratext_dict[p].split()) for p in paras]
        l10 = len([x for x in lens if x < 10])
        l20 = len([x for x in lens if 10 <= x < 20])
        l30 = len([x for x in lens if 20 <= x < 30])
        l40 = len([x for x in lens if 30 <= x < 40])
        l50 = len([x for x in lens if 40 <= x])
        arts.append(p)
        curr_stat = [n, top_k, mean_top_k, std_top_k, min_top_k, max_top_k, hier_k, mean_hier_k, std_hier_k,
                     min_hier_k, max_hier_k, l10, l20, l30, l40, l50]
        stats.append(curr_stat)
        c += 1
        if i > 0 and c >= i:
            break
    print('Article\tN\ttop_k\tmean_k\tstd_k\tmin_k\tmax_k\thier_k\tmean_k\tstd_k\tmin_k\tmax_k\tl10\tl20\tl30\tl40\tl50')
    for i, d in enumerate(stats):
        print(arts[i]+'\t'.join([str(dd) for dd in d]))
    return arts, stats

articles, stats = trec_stats(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[6]))
with open(sys.argv[5]+'/articles.json') as f:
    json.dump(articles, f)
np.save(sys.argv[5]+'/stats', np.array(stats))