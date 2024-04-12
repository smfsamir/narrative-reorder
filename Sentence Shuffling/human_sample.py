import polars as pl
import numpy as np

file = "/Users/goose/Documents/CPSC 448/Sentence Shuffling/shuffled_sent_data.jsonl"

df = pl.read_ndjson(file)

sample = df.sample(50, with_replacement=False)

first = sample[:30]
second = sample[-30:]

first.write_ndjson("first sample.jsonl")
first.write_ndjson("second sample.jsonl")
