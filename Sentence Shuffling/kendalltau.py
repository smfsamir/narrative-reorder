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

third = pl.read_csv("anno_data/third_answers.csv")

third = third.with_columns(
    pl.col('anno_label').str.json_decode().alias('anno_label'),
    pl.col('label').str.json_decode().alias('label')
)

third = third.with_columns(
    pl.struct(['anno_label']).map_elements(lambda x: len(x['anno_label'])).alias('anno_length'),
    pl.struct(['label']).map_elements(lambda x: len(x['label'])).alias('label_length')
)

third = third.with_columns(
    pl.struct(['anno_label', 'label']).map_elements(lambda x: normalised_kendall_tau_distance(x['anno_label'], x['label'])).alias('kendall_tau_distance')
)

anno_agreement = pl.read_csv('anno_data/anno_agreement.csv')

anno_agreement = anno_agreement.with_columns(
    pl.col('first_anno').str.json_decode().alias('first_anno'),
    pl.col('second_anno').str.json_decode().alias('second_anno'),
    pl.col('third_anno').str.json_decode().alias('third_anno')
)

anno_agreement = anno_agreement.with_columns(
    pl.struct(['first_anno', 'second_anno']).map_elements(lambda x: normalised_kendall_tau_distance(x['first_anno'], x['second_anno'])).alias('first_second_KTD'),
    pl.struct(['first_anno', 'third_anno']).map_elements(lambda x: normalised_kendall_tau_distance(x['first_anno'], x['third_anno'])).alias('first_third_KTD'),
    pl.struct(['third_anno', 'second_anno']).map_elements(lambda x: normalised_kendall_tau_distance(x['third_anno'], x['second_anno'])).alias('third_second_KTD')
)



""" ## print(first)
print(sp.stats.spearmanr(first['label_length'], first['kendall_tau_distance']))
print(first.mean())

## print(second.head())
print(sp.stats.spearmanr(second['label_length'], second['kendall_tau_distance']))
print(second.mean())

## print(third.head())
print(sp.stats.spearmanr(third['label_length'], third['kendall_tau_distance']))
print(third.mean()) """

print(anno_agreement.head())
print(anno_agreement.mean())

""" first.write_ndjson("/Users/goose/Documents/Projects/CPSC 448/narrative-reorder/Sentence Shuffling/anno_data/first.csv")
second.write_ndjson("/Users/goose/Documents/Projects/CPSC 448/narrative-reorder/Sentence Shuffling/anno_data/second.csv")
third.write_ndjson("/Users/goose/Documents/Projects/CPSC 448/narrative-reorder/Sentence Shuffling/anno_data/third.csv")  """
anno_agreement.write_ndjson("/Users/goose/Documents/Projects/CPSC 448/narrative-reorder/Sentence Shuffling/anno_data/anno_agreement_KTD.csv")