import numpy as np
import json
import polars as pl

def convert_to_string(lst_sentences: list[str]) -> str:
    num = 1
    prompt = ""

    for sentence in lst_sentences:
        prompt = prompt + str(num) + ". " + sentence + " "
        num += 1
    
    return prompt


dataset = [json.loads(line) for line in open("shuffled_sent_data.jsonl")]
labels = [d['label'] for d in dataset]
sentence_lists = [convert_to_string(d['list_of_sentences']) for d in dataset]

frame = pl.DataFrame({
    "label": labels,
    "sentence_lists": sentence_lists
})

sample = np.random.choice(len(frame), size=20, replace=False)
first_sample_inds = sample[:15]
""" second_sample_inds = sample[20:50]
third_sample_inds = sample[40:-30] """

frame_first = frame.with_row_count().filter(pl.col("row_nr").is_in(first_sample_inds))
""" frame_second = frame.with_row_count().filter(pl.col("row_nr").is_in(second_sample_inds))
frame_third = frame.with_row_count().filter(pl.col("row_nr").is_in(second_sample_inds)) """

frame_first.write_csv("additional sample.csv", separator=",")
""" frame_second.write_csv("second sample.csv", separator=",")
frame_third.write_csv("third sample.csv", separator=",") """