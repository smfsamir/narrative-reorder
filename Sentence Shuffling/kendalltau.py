import numpy as np
import scipy as sp
import polars as pl

def normalised_kendall_tau_distance(values1, values2):
    """Compute the Kendall tau distance."""
    n = len(values1)
    if len(values2) != n:
        return None
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))


## process first sample
first = pl.read_csv("anno_data/first_answers.csv")

## extract lists from annotation/label strings
first = first.with_columns(
    pl.col('anno_label').str.json_decode().alias('anno_label'),
    pl.col('label').str.json_decode().alias('label')
)

## compute length of annotation/label for error checking
first = first.with_columns(
    pl.struct(['anno_label']).map_elements(lambda x: len(x['anno_label'])).alias('anno_length'),
    pl.struct(['label']).map_elements(lambda x: len(x['label'])).alias('label_length')
)

## calculate kendall tau distance
first = first.with_columns(
    pl.struct(['anno_label', 'label']).map_elements(lambda x: normalised_kendall_tau_distance(x['anno_label'], x['label'])).alias('kendall_tau_distance'),
)

## spearman correlation between length and kendall tau distance???
first = first.with_columns(
    pl.struct(['label_length', 'kendall_tau_distance']).map_elements(lambda x: sp.stats.spearmanr(x['label_length'], x['kendall_tau_distance'])).alias('length_corr')
)


second = pl.read_csv("anno_data/second_answers.csv")

second = second.with_columns(
    pl.col('anno_label').str.json_decode().alias('anno_label'),
    pl.col('label').str.json_decode().alias('label')
)

second = second.with_columns(
    pl.struct(['anno_label']).map_elements(lambda x: len(x['anno_label'])).alias('anno_length'),
    pl.struct(['label']).map_elements(lambda x: len(x['label'])).alias('label_length')
)

second = second.with_columns(
    pl.struct(['anno_label', 'label']).map_elements(lambda x: normalised_kendall_tau_distance(x['anno_label'], x['label'])).alias('kendall_tau_distance')
)

print(first)
print(second)