#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import statsmodels.stats.api as sms
from granatum_sdk import Granatum
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture as GM
import statistics as s
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

min_dist = 2.0
min_zscore = 2.0

def confint(X, alpha=0.05):
    resultdict = {}
    meanbounds = sms.DescrStatsW(X).tconfint_mean(alpha=alpha)
    resultdict["low"] = meanbounds[0]
    resultdict["high"] = meanbounds[1]
    resultdict["n"] = len(X)
    return resultdict


def dist(int1, int2):
    if int1["low"] >= int2["high"]:
        return int1["low"] - int2["high"]
    if int2["low"] >= int1["high"]:
        return int2["low"] - int1["high"]
    return 0.0


# return hash of labels associated to its data
def trygmonvector(gm, X):
    vectors = gm.predict(np.array(X).reshape(-1, 1))
    inv_map = {}
    for i, v in enumerate(vectors):
        inv_map[v] = inv_map.get(v, []) + [X[i]]
    return inv_map


# First try two mixtures
# Return: {"data": X, "gm": GM, low_means: [], high_means: [], n: []}
def one_or_two_mixtures(X, alpha=0.05, min_dist=min_dist):
    column = np.array(X).reshape(-1, 1)
    gm = GM(n_components=2).fit(column)
    inv_map = trygmonvector(gm, X)
    mean = s.mean(X)
    std = s.stdev(X)

    if len(inv_map) <= 1 or len(inv_map[0]) < 3 or len(inv_map[1]) < 3:
        gm = GM(n_components=1).fit(column)
        mi = confint(X)
        return {"data": X, "mean": mean, "std": std, "gm": gm, "low_means": [mi["low"]], "high_means": [mi["high"]], "n": [len(X)]}

    mi1 = confint(inv_map[0], alpha=alpha)
    mi2 = confint(inv_map[1], alpha=alpha)
    # zscore1 = abs(s.mean(inv_map[0])-s.mean(inv_map[1]))/(s.stdev(inv_map[1])+1e-16)
    # zscore2 = abs(s.mean(inv_map[1])-s.mean(inv_map[0]))/(s.stdev(inv_map[0])+1e-16)
    if dist(mi1, mi2) <= min_dist:
        gm = GM(n_components=1).fit(column)
        mi = confint(X)
        result = {"data": X, "mean": mean, "std": std, "gm": gm, "low_means": [mi["low"]], "high_means": [mi["high"]], "n": [len(X)]}
    elif mi1["low"] < mi2["low"]:
        result = {"data": X, "mean": mean, "std": std, "gm": gm, "label_order": [0, 1], "low_means": [mi1["low"], mi2["low"]], "high_means": [mi1["high"], mi2["high"]], "n": [mi1["n"], mi2["n"]]}
    else:
        result = {"data": X, "mean": mean, "std": std, "gm": gm, "label_order": [1, 0], "low_means": [mi2["low"], mi1["low"]], "high_means": [mi2["high"], mi1["high"]], "n": [mi2["n"], mi1["n"]]}
    return result


def gen_expression_count(X1, X2, min_zscore=min_zscore):
    # print(((np.asarray(X1["data"]) - X2["mean"]) / (X2["std"] + 1e-16)))
    return (((np.asarray(X1["data"]) - X2["mean"]) / (X2["std"] + 1e-16)) > min_zscore).sum() / X1["n"][0]


# Just compute X1s expression level given X2
def compute_percent_expression(X1, X2, min_zscore=min_zscore):
    if len(X2["low_means"]) > 1:
        inv_map = trygmonvector(X2["gm"], X1["data"])
        if X2["label_order"][1] in inv_map:
            expressed_num = len(inv_map[X2["label_order"][1]])
        else:
            expressed_num = 0
        if X2["label_order"][0] in inv_map:
            unexpressed_num = len(inv_map[X2["label_order"][0]])
        else:
            unexpressed_num = 0
        result_percent_expressed = expressed_num/(expressed_num+unexpressed_num)
    elif len(X1["low_means"]) > 1:
        result_percent_expressed = X1["n"][1]/(X1["n"][0]+X1["n"][1])
    else:
        if X1["low_means"][0] > X2["high_means"][0]:
            result_percent_expressed = gen_expression_count(X1, X2, min_zscore=min_zscore)
        elif X1["high_means"][0] < X2["low_means"][0]:
            result_percent_expressed = gen_expression_count(X1, X2, min_zscore=min_zscore)
        else:
            result_percent_expressed = 0.0
    return result_percent_expressed*100.0


# Note that X2 provies the statistic for saying "on" or "off" if it has two states
def compute_percent_diff(X1, X2, min_zscore=min_zscore):
    if(len(X2["low_means"]) > 1):
        source_percent_expressed = X2["n"][1]/(X2["n"][0]+X2["n"][1])
        inv_map = trygmonvector(X2["gm"], X1["data"])
        if X2["label_order"][1] in inv_map:
            expressed_num = len(inv_map[X2["label_order"][1]])
        else:
            expressed_num = 0
        if X2["label_order"][0] in inv_map:
            unexpressed_num = len(inv_map[X2["label_order"][0]])
        else:
            unexpressed_num = 0
        result_percent_expressed = expressed_num/(expressed_num+unexpressed_num)
    elif(len(X1["low_means"]) > 1):
        result_percent_expressed = X1["n"][1]/(X1["n"][0]+X1["n"][1])
        inv_map = trygmonvector(X1["gm"], X2["data"])
        if X1["label_order"][1] in inv_map:
            expressed_matching = len(inv_map[X1["label_order"][1]])
        else:
            expressed_matching = 0.0
        if X1["label_order"][0] in inv_map:
            unexpressed_matching = len(inv_map[X1["label_order"][0]])
        else:
            unexpressed_matching = 0.0
        if unexpressed_matching > expressed_matching:
            source_percent_expressed = 0.0
        else:
            source_percent_expressed = 1.0
    else:
        if X1["low_means"][0] > X2["high_means"][0]:
            source_percent_expressed = gen_expression_count(X2, X1, min_zscore=min_zscore)
            result_percent_expressed = gen_expression_count(X1, X2, min_zscore=min_zscore)
        elif X1["high_means"][0] < X2["low_means"][0]:
            source_percent_expressed = gen_expression_count(X2, X1, min_zscore=min_zscore)
            result_percent_expressed = gen_expression_count(X1, X2, min_zscore=min_zscore)
        else:
            result_percent_expressed = 0.0
            source_percent_expressed = 0.0

    percent_diff = (result_percent_expressed - source_percent_expressed)*100.0
    return percent_diff


def compref(gene, row, colnames, inv_map, inv_map_rest, alpha, min_dist, min_zscore):
    base_statistic = one_or_two_mixtures(row, alpha=alpha, min_dist=min_dist)
    cluster_statistics = {}
    for cluster, v in inv_map.items():
        cluster_statistics[cluster] = compute_percent_expression(one_or_two_mixtures(row[v].tolist(), alpha=alpha, min_dist=min_dist), base_statistic)
    cluster_rest_statistics = {}
    for cluster, v in inv_map_rest.items():
        cluster_rest_statistics[cluster] = compute_percent_expression(one_or_two_mixtures(row[v].tolist(), alpha=alpha, min_dist=min_dist), base_statistic)
    result = pd.DataFrame(index=[gene], columns=colnames)
    for cnamei, sti in cluster_statistics.items():
        for cnamej, stj in cluster_statistics.items():
            if cnamei != cnamej:
                result["{} vs {}".format(cnamei, cnamej)][gene] = sti-stj
    for cnamei, sti in cluster_rest_statistics.items():
        stself = cluster_statistics[cnamei]
        result["{} vs rest".format(cnamei)][gene] = stself - sti
    return result


def comp(gene, row, colnames, inv_map, inv_map_rest, alpha, min_dist, min_zscore):
    cluster_statistics = {}
    for cluster, v in inv_map.items():
        cluster_statistics[cluster] = one_or_two_mixtures(row[v].tolist(), alpha=alpha, min_dist=min_dist)
    cluster_rest_statistics = {}
    for cluster, v in inv_map_rest.items():
        cluster_rest_statistics[cluster] = one_or_two_mixtures(row[v].tolist(), alpha=alpha, min_dist=min_dist)
    result = pd.DataFrame(index=[gene], columns=colnames)
    for cnamei, sti in cluster_statistics.items():
        for cnamej, stj in cluster_statistics.items():
            if cnamei != cnamej:
                result["{} vs {}".format(cnamei, cnamej)][gene] = compute_percent_diff(sti, stj, min_zscore=min_zscore)
    for cnamei, sti in cluster_rest_statistics.items():
        stself = cluster_statistics[cnamei]
        result["{} vs rest".format(cnamei)][gene] = compute_percent_diff(stself, sti, min_zscore=min_zscore)
    return result


def main():
    tic = time.perf_counter()

    gn = Granatum()

    assay = gn.pandas_from_assay(gn.get_import('assay'))
    # Groups is {"cell":"cluster}
    groups = gn.get_import('groups')

    certainty = gn.get_arg('certainty')
    alpha = 1 - certainty/100.0
    
    min_zscore = gn.get_arg('min_zscore')
    min_dist = gn.get_arg('min_dist')

    # Likely we want to filter genes before we get started, namely if we cannot create a good statistic
    norms_df = assay.apply(np.linalg.norm, axis=1)
    assay = assay.loc[norms_df.T >= min_dist, :]

    inv_map = {}
    inv_map_rest = {}
    for k, v in groups.items():
        inv_map[v] = inv_map.get(v, []) + [k]
        clist = inv_map_rest.get(v, list(assay.columns))
        clist.remove(k)
        inv_map_rest[v] = clist
    # Inv map is {"cluster": ["cell"]}
    print("Completed setup", flush=True)
    
    cols = list(inv_map.keys())
    genes = assay.index.tolist()
    
    colnames = []
    for coli in cols:
        for colj in cols:
            if coli != colj:
                colnames.append("{} vs {}".format(coli, colj))
    for coli in cols:
        colnames.append("{} vs rest".format(coli))

    # result = pd.DataFrame(index=genes, columns=colnames)
    # result.fillna(0)

    # Instead of scoring into a dataframe, let's analyze each statistically
    # Dict (gene) of dict (cluster) of dict (statistics)
    # { "gene_name" : { "cluster_name" : { statistics data } }}
    # Export would be percentage more/less expressed in "on" state
    # For example gene "XIST" expresses at least 20% more in cluster 1 vs cluster 4 with 95% certainty
    gene_count = 0;
    total_genes = len(assay.index)
    print("Executing parallel for {} genes".format(total_genes), flush=True)

    results = Parallel(n_jobs=math.floor(multiprocessing.cpu_count()*9/10))(delayed(compref)(gene, assay.loc[gene, :], colnames, inv_map, inv_map_rest, alpha, min_dist, min_zscore) for gene in tqdm(list(assay.index)))
    result = pd.concat(results, axis=0)

    gn.export_statically(gn.assay_from_pandas(result.T), 'Differential expression sets')
    gn.export(result.to_csv(), 'differential_gene_sets.csv', kind='raw', meta=None, raw=True)

    toc = time.perf_counter()
    time_passed = round(toc - tic, 2)

    timing = "* Finished differential expression sets step in {} seconds*".format(time_passed)
    gn.add_result(timing, "markdown")

    gn.commit()


if __name__ == '__main__':
    main()
