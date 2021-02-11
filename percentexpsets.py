#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import statsmodels.stats.api as sms
from granatum_sdk import Granatum
from scipy.stats import poisson
from scipy.optimize import least_squares
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


def trygmonvector(gm, X):   # return hash of labels associated to its data
    vectors = gm.predict(np.array(X).reshape(-1, 1))
    inv_map = {}
    for i, v in enumerate(vectors):
        inv_map[v] = inv_map.get(v, []) + [X[i]]
    return inv_map


def model1(t, coeffs):
    lamb1 = coeffs[0]
    eval1 = poisson.pmf(np.floor(t), lamb1)
    eval2 = poisson.pmf(np.ceil(t), lamb1)
    return eval1 + (t - np.floor(t)) * (eval2 - eval1)


def residuals1(coeffs, y, t):
    return y - model1(t, coeffs)


def model2(t, coeffs):
    a = coeffs[0]
    lamb1 = coeffs[1]
    lamb2 = coeffs[2]
    return (a * model1(t, [lamb1]) + model1(t, [lamb2])) / (1.0 + a)


def residuals2(coeffs, y, t):
    return y - model2(t, coeffs)


def model3_fix_params(lamb1, lamb2):
    def model3(t, coeffs):
        a = coeffs[0]
        return model2(t, [a, lamb1, lamb2])

    return model3


def residuals3_fix_params(lamb1, lamb2):
    model3 = model3_fix_params(lamb1, lamb2)

    def residuals3(coeffs, y, t):
        return y - model3(t, coeffs)

    return residuals3


def fit_data_two_poissons(X, initial):
    entries, bins = np.histogram(X, bins="auto", density=True)
    bins = (bins[1:] + bins[:-1]) / 2
    parameters = least_squares(residuals2, initial, args=(entries, bins))
    return parameters


def fit_data_one_poisson(X, initial):
    entries, bins = np.histogram(X, bins="auto", density=True)
    bins = (bins[1:] + bins[:-1]) / 2
    parameters = least_squares(residuals1, initial, args=(entries, bins))
    return parameters


def fit_poissons_fixed_means(X, lamb1, lamb2):
    residuals3 = residuals3_fix_params(lamb1, lamb2)
    entries, bins = np.histogram(X, bins="auto", density=True)
    bins = (bins[1:] + bins[:-1]) / 2
    parameters = least_squares(residuals3, [0.5], args=(entries, bins))
    parameters.x = np.append(parameters.x, [lamb1, lamb2])
    return parameters


def one_or_two_mixtures(X, alpha=0.05, min_dist=0.2, min_zscore=2):
    column = np.array(X).reshape(-1, 1)
    gm = GM(n_components=2).fit(column)
    inv_map = trygmonvector(gm, X)
    mean = np.mean(X)
    std = np.std(X)

    if len(inv_map) <= 1 or len(inv_map[0]) < 3 or len(inv_map[1]) < 3:
        gm = GM(n_components=1).fit(column)
        mi = confint(X)
        return {"data": X, "mean": mean, "std": std, "gm": gm, "low_means": [mi["low"]], "high_means": [mi["high"]],
                "n": [len(X)]}

    mi1 = confint(inv_map[0], alpha=alpha)
    mi2 = confint(inv_map[1], alpha=alpha)
    if dist(mi1, mi2) <= min_dist or abs(gm.means_[1][0] - gm.means_[0][0]) / (max(gm.covariances_)[0][0]) < min_zscore:
        gm = GM(n_components=1).fit(column)
        mi = confint(X)
        result = {"data": X, "mean": mean, "std": std, "gm": gm, "low_means": [mi["low"]], "high_means": [mi["high"]],
                  "n": [len(X)]}
    elif mi1["low"] < mi2["low"]:
        result = {"data": X, "mean": mean, "std": std, "gm": gm, "label_order": [0, 1],
                  "low_means": [mi1["low"], mi2["low"]], "high_means": [mi1["high"], mi2["high"]],
                  "n": [mi1["n"], mi2["n"]]}
    else:
        result = {"data": X, "mean": mean, "std": std, "gm": gm, "label_order": [1, 0],
                  "low_means": [mi2["low"], mi1["low"]], "high_means": [mi2["high"], mi1["high"]],
                  "n": [mi2["n"], mi1["n"]]}
    return result

# First transform X into log(X)+c such that it does not go below 0, X is a list
def fit_poissons(X, alpha=0.05, min_dist=0.2, min_zscore=2):
    if np.mean(X) < 5:  # Can't really form a good statistic
        return {"n": 1, "coeffs": [np.mean(X), np.std(X)]}
    shift = np.min(X) - 1  # Needed later to shift back
    Xarr = np.log(X - shift)
    res = one_or_two_mixtures(Xarr.tolist(), alpha=0.05, min_dist=min_dist, min_zscore=min_zscore)
    numcomponents = len(res["low_means"])
    if numcomponents == 2:
        mean1 = 0.5 * (res["low_means"][0] + res["high_means"][0])
        mean2 = 0.5 * (res["low_means"][1] + res["high_means"][1])
        mean1 = np.exp(mean1) + shift
        mean2 = np.exp(mean2) + shift
        sz1 = res["n"][0]
        sz2 = res["n"][1]
        alpha = sz1 / sz2
        # Now optimize with estimates
        # coeffs = fit_data_two_poissons(X, [alpha, mean1, mean2])
        coeffs_fm = fit_poissons_fixed_means(X, mean1, mean2)
        print("Optimality fm = {}".format(coeffs_fm.cost))
        coeffs_2 = fit_data_two_poissons(X, [alpha, mean1, mean2])
        print("Optimality 2 = {}".format(coeffs_2.cost))
        coeffs_1 = fit_data_one_poisson(X, [np.mean(X)])
        print("Optimality 1 = {}".format(coeffs_1.cost))
        if coeffs_2.cost < coeffs_fm.cost:
            coeffs = coeffs_2
        else:
            coeffs = coeffs_fm
        if coeffs.x[0] > 0.0 and 2 * coeffs.cost < coeffs_1.cost:
            return {"n": 2, "coeffs": coeffs}

    print("Only have one!")
    Xarr = np.array(X)
    mean1 = np.mean(Xarr)
    mean2 = mean1 + min_zscore * np.sqrt(mean1)
    mean1 = np.mean(Xarr[Xarr < mean2 - np.sqrt(mean2) / 2.0])

    coeffs = fit_poissons_fixed_means(X, mean1, mean2)
    print("Alpha = {}".format(coeffs.x[0]))
    # coeffs = fit_data_one_poisson(X, [mean1])
    return {"n": 2, "coeffs": coeffs}


def percent_upregulation(a):
    return 100.0*(1.0/(a+1.0))


def upregulation_from_gaussian(X, mean, std, min_zscore=2):
    X = np.array(X)
    return 100.0*np.size(X[X-mean > 0]) / np.size(X)


def get_poisson_weight_with_statistic(row, params):   # Returns a
    coeffs_fm = fit_poissons_fixed_means(row, params["coeffs"].x[1], params["coeffs"].x[2]) # Use the statistic means
    print("Optimality fm = {}".format(coeffs_fm.cost))
    return coeffs_fm.x[0]


def predict_row(row, params, min_zscore = 2):
    if params["n"] == 1:
        return upregulation_from_gaussian(row, params["coeffs"][0], params["coeffs"][1], min_zscore=min_zscore)

    a = get_poisson_weight_with_statistic(row, params)
    params["coeffs"].x[0] = a                   # Update params with fitted a, this may side-effect
    return percent_upregulation(a)


def fit_row(row, alpha=0.05, min_dist=0.2, min_zscore = 2):  # Return the fitted params
    params = fit_poissons(row, alpha=alpha, min_dist=min_dist, min_zscore=min_zscore)
    return params


# Main computation point
def compref(gene, row, colnames, inv_map, inv_map_rest, alpha, min_dist, min_zscore):
    base_statistic = fit_row(row.tolist(), alpha=alpha, min_dist=min_dist, min_zscore=min_zscore)
    cluster_statistics = {}
    for cluster, v in inv_map.items():
        cluster_statistics[cluster] = predict_row(row[v].tolist(), base_statistic, min_zscore=min_zscore)
    cluster_rest_statistics = {}
    for cluster, v in inv_map_rest.items():
        cluster_rest_statistics[cluster] = predict_row(row[v].tolist(), base_statistic)
    result = pd.DataFrame(index=[gene], columns=colnames)
    for cnamei, sti in cluster_statistics.items():
        for cnamej, stj in cluster_statistics.items():
            if cnamei != cnamej:
                result["{} vs {}".format(cnamei, cnamej)][gene] = sti-stj
    if len(colnames) > 2:
        for cnamei, sti in cluster_rest_statistics.items():
            stself = cluster_statistics[cnamei]
            result["{} vs rest".format(cnamei)][gene] = stself - sti
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

    colnames = []
    for coli in cols:
        for colj in cols:
            if coli != colj:
                colnames.append("{} vs {}".format(coli, colj))
    for coli in cols:
        colnames.append("{} vs rest".format(coli))

    # Instead of scoring into a dataframe, let's analyze each statistically
    # Dict (gene) of dict (cluster) of dict (statistics)
    # { "gene_name" : { "cluster_name" : { statistics data } }}
    # Export would be percentage more/less expressed in "on" state
    # For example gene "XIST" expresses at least 20% more in cluster 1 vs cluster 4 with 95% certainty
    total_genes = len(assay.index)
    print("Executing parallel for {} genes".format(total_genes), flush=True)

    results = Parallel(n_jobs=math.floor(multiprocessing.cpu_count()*2*9/10))(delayed(compref)(gene, assay.loc[gene, :], colnames, inv_map, inv_map_rest, alpha, min_dist, min_zscore) for gene in tqdm(list(assay.index)))
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
