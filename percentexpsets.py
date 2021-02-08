#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import statsmodels.stats.api as sms
from granatum_sdk import Granatum
from scipy.optimize import curve_fit

def range_check(x, y):
    if(x <= 0.0 and y <= 0.0):
        return max(x, y)
    if (x >= 0.0 and y >= 0.0):
        return min(x, y)
    return 0.0

def main():
    tic = time.perf_counter()

    gn = Granatum()

    assay = gn.pandas_from_assay(gn.get_import('assay'))
    groups = gn.get_import('groups')

    certainty = gn.get_arg('certainty')

    inv_map = {}
    for k, v in groups.items():
        inv_map[v] = inv_map.get(v, []) + [k]

    # Instead of scoring into a dataframe, let's analyze each statistically
    # Dict (gene) of dict (cluster) of dict (statistics)
    # { "gene_name" : { "cluster_name" : { statistics data } }}
    # Export would be percentage more/less expressed in "on" state
    # For example gene "XIST" expresses at least 20% more in cluster 1 vs cluster 4 with 95% certainty
    low_mean_dfs = []
    high_mean_dfs = []
    mean_dfs = []
    std_dfs = []
    colnames = []
    for k, v in inv_map.items():
        group_values = assay.loc[:, v] # All genes for group of cells
        lowbound_clust = {}
        highbound_clust = {}
        for index, row in group_values.iterrows():
            meanbounds = sms.DescrStatsW(row).tconfint_mean()
            lowbound_clust[index] = meanbounds[0]
            highbound_clust[index] = meanbounds[1]
        low_mean_dfs.append(pd.DataFrame.from_dict(lowbound_clust, orient="index", columns=[k]))
        high_mean_dfs.append(pd.DataFrame.from_dict(highbound_clust, orient="index", columns=[k]))
        mean_dfs.append(group_values.mean(axis=1))
        std_dfs.append(group_values.std(axis=1))
        colnames.append(k)
    mean_df = pd.concat(mean_dfs, axis=1)
    mean_df.columns = colnames
    low_mean_df = pd.concat(low_mean_dfs, axis=1)
    low_mean_df.columns = colnames
    high_mean_df = pd.concat(high_mean_dfs, axis=1)
    high_mean_df.columns = colnames
    std_df = pd.concat(std_dfs, axis=1)
    std_df.columns = colnames
    print(std_df)
    minvalues = std_df.min(axis=1).to_frame()
    minvalues.columns=["min"]
    print("Minvalues>>")
    print(minvalues, flush=True)
    genes_below_min = list((minvalues[minvalues["min"]<min_expression_variation]).index)
    print("{} out of {}".format(len(genes_below_min), len(minvalues.index)), flush=True)
    mean_df = mean_df.drop(genes_below_min, axis=0)
    low_mean_df = low_mean_df.drop(genes_below_min, axis=0)
    high_mean_df = high_mean_df.drop(genes_below_min, axis=0)
    std_df = std_df.drop(genes_below_min, axis=0)
    assay = assay.drop(genes_below_min, axis=0)
    print("Filtered assay to get {} columns by {} rows".format(len(assay.columns), len(assay.index)), flush=True)

    mean_rest_dfs = []
    std_rest_dfs = []
    colnames = []
    for k, v in inv_map.items():
        rest_v = list(set(list(assay.columns)).difference(set(v)))
        mean_rest_dfs.append(assay.loc[:, rest_v].mean(axis=1))
        std_rest_dfs.append(assay.loc[:, rest_v].std(axis=1))
        colnames.append(k)
    mean_rest_df = pd.concat(mean_rest_dfs, axis=1)
    mean_rest_df.columns = colnames
    std_rest_df = pd.concat(std_rest_dfs, axis=1)
    std_rest_df.columns = colnames

    zscore_dfs = []
    cols = colnames
    colnames = []
    for coli in cols:
        for colj in cols:
            if coli != colj:
                # Here we should check significance
                # Fetch most realistic mean comparison set, what is smallest difference between two ranges
                mean_diff_overlap_low_high = (low_mean_df[coli]-high_mean_df[colj])
                mean_diff_overlap_high_low = (high_mean_df[coli]-low_mean_df[colj])
                diff_df = mean_diff_overlap_low_high.combine(mean_diff_overlap_high_low, range_check)

                zscore_dfs.append((diff_df/(std_df[colj]+std_df[coli]/4)).fillna(0).clip(-max_zscore, max_zscore))
                colnames.append("{} vs {}".format(coli, colj)) 
    for coli in cols:
        zscore_dfs.append(((mean_df[coli]-mean_rest_df[colj])/(std_rest_df[colj]+std_rest_df[coli]/4)).fillna(0).clip(-max_zscore, max_zscore))
        colnames.append("{} vs rest".format(coli)) 

    zscore_df = pd.concat(zscore_dfs, axis=1)
    zscore_df.columns = colnames
    norms_df = zscore_df.apply(np.linalg.norm, axis=1)
    colsmatching = norms_df.T[(norms_df.T >= min_zscore)].index.values
    return_df = zscore_df.T[colsmatching]
    gn.export_statically(gn.assay_from_pandas(return_df), 'Differential expression sets')
    gn.export(return_df.T.to_csv(), 'differential_gene_sets.csv', kind='raw', meta=None, raw=True)

    toc = time.perf_counter()
    time_passed = round(toc - tic, 2)

    timing = "* Finished differential expression sets step in {} seconds*".format(time_passed)
    gn.add_result(timing, "markdown")

    gn.commit()


if __name__ == '__main__':
    main()
