# BROKEN DUE TO FILE REORDERING; TODO

import polars as pl
import pathlib
import os

boolq_path = "/Users/goose/Documents/CPSC 448/filter-data/glue_filtered/boolq.train.json"
cola_path = "/Users/goose/Documents/CPSC 448/filter-data/glue_filtered/cola.train.json"
mnli_path = "/Users/goose/Documents/CPSC 448/filter-data/glue_filtered/mnli.train.json"
mrpc_path = "/Users/goose/Documents/CPSC 448/filter-data/glue_filtered/mrpc.train.json"
multirc_path = "/Users/goose/Documents/CPSC 448/filter-data/glue_filtered/multirc.train.json"
qnli_path = "/Users/goose/Documents/CPSC 448/filter-data/glue_filtered/qnli.train.json"
qqp_path = "/Users/goose/Documents/CPSC 448/filter-data/glue_filtered/qqp.train.json" 
rte_path = "/Users/goose/Documents/CPSC 448/filter-data/glue_filtered/rte.train.json"
sst2_path = "/Users/goose/Documents/CPSC 448/filter-data/glue_filtered/sst2.train.json"
wsc_path = "/Users/goose/Documents/CPSC 448/filter-data/glue_filtered/wsc.train.json" ##TODO


glue_dfs = []

""" df_schema = pl.DataFrame(
    {
        "id": pl.Int16,
        "dataset": pl.String,
        "subset": pl.String,
        "test": pl.String,
        "input_question": pl.String,
        "ground_truth": pl.String,
        "alexa": pl.String
    }
) """

### parse boolq set ###
boolq_df = pl.read_ndjson(boolq_path)
boolq_sample = boolq_df.sample(2, with_replacement=False)
    
## Convert input question to string
boolq_input_question_str = boolq_sample \
    .select(pl.format("{question: {}, passage: {}, Does the passage answer the question?}",
                        pl.col("question"),
                        pl.col("passage")))

## Convert ground_truth to string
boolq_ground_truth_str = boolq_sample.select(pl.format("{}",
                                                     pl.col("label")))

boolq_temp = pl.DataFrame(
    {
    "id": boolq_sample.select("idx"),
    "dataset": ["superglue"] * len(boolq_sample),
    "subset": ["boolq"] * len(boolq_sample),
    "test": ["null"] * len(boolq_sample),
    "input_question": boolq_input_question_str,
    "ground_truth": boolq_ground_truth_str,
    "alexa": ["TODO"] * len(boolq_sample)
    }
)
    
print(boolq_temp)
glue_dfs.append(boolq_temp)

### parse cola set ###
cola_df = pl.read_ndjson(cola_path)
cola_sample = cola_df.sample(2, with_replacement=False)
    
## Convert ground_truth to string
cola_ground_truth_str = cola_sample.select(pl.format("{}",
                                                     pl.col("label")))

cola_temp = pl.DataFrame(
    {
    "id": cola_sample.select("idx"),
    "dataset": ["glue"] * len(cola_sample),
    "subset": ["cola"] * len(cola_sample),
    "test": ["null"] * len(cola_sample),
    "input_question": cola_sample.select("sentence"),
    "ground_truth": cola_ground_truth_str,
    "alexa": ["TODO"] * len(cola_sample)
    }
)
    
print(cola_temp)
glue_dfs.append(cola_temp)


### parse mnli set ###
mnli_df = pl.read_ndjson(mnli_path)
mnli_sample = mnli_df.sample(2, with_replacement=False)

## Convert input question to string
mnli_input_question_str = mnli_sample \
    .select(pl.format("{premise: {}, hypothesis: {}, Is the hypothesis entailment, contradiction, or neutral?}",
                        pl.col("premise"),
                        pl.col("hypothesis")))

## Convert ground_truth to string
mnli_ground_truth_str = mnli_sample.select(pl.format("{}",
                                                     pl.col("label")))

mnli_temp = pl.DataFrame(
    {
    "id": mnli_sample.select("idx"),
    "dataset": ["glue"] * len(mnli_sample),
    "subset": ["mnli"] * len(mnli_sample),
    "test": ["null"] * len(mnli_sample),
    "input_question": mnli_input_question_str,
    "ground_truth": mnli_ground_truth_str,
    "alexa": ["TODO"] * len(mnli_sample)
    }
)
    
print(mnli_temp)
glue_dfs.append(mnli_temp)

### parse mrpc set ###
mrpc_df = pl.read_ndjson(mrpc_path)
mrpc_sample = mrpc_df.sample(2, with_replacement=False)
    
## Convert input question to string
mrpc_input_question_str = mrpc_sample \
    .select(pl.format("{sentence 1: {}, sentence 2: {}, Are the two sentences semantically equivalent?}",
                        pl.col("sentence1"),
                        pl.col("sentence2")))

## Convert ground_truth to string
mrpc_ground_truth_str = mrpc_sample.select(pl.format("{}",
                                                     pl.col("label")))

mrpc_temp = pl.DataFrame(
    {
    "id": mrpc_sample.select("idx"),
    "dataset": ["glue"] * len(mrpc_sample),
    "subset": ["mrpc"] * len(mrpc_sample),
    "test": ["null"] * len(mrpc_sample),
    "input_question": mrpc_input_question_str,
    "ground_truth": mrpc_ground_truth_str,
    "alexa": ["TODO"] * len(mrpc_sample)
    }
)
    
print(mrpc_temp)
glue_dfs.append(mrpc_temp)

### parse multirc set ###
multirc_df = pl.read_ndjson(multirc_path)
multirc_sample = multirc_df.sample(2, with_replacement=False)
    
## Convert input question to string
multirc_input_question_str = multirc_sample \
    .select(pl.format("{paragraph: {}, question: {}, answer: {}, Is this the correct answer?}",
                        pl.col("paragraph"),
                        pl.col("question"),
                        pl.col("answer")))

## Convert ground_truth to string
multirc_ground_truth_str = multirc_sample.select(pl.format("{}",
                                                     pl.col("label")))

multirc_temp = pl.DataFrame(
    {
    "id": multirc_sample.select("question").cast(pl.Int64, strict=False),
    "dataset": ["superglue"] * len(multirc_sample),
    "subset": ["multirc"] * len(multirc_sample),
    "test": ["null"] * len(multirc_sample),
    "input_question": multirc_input_question_str,
    "ground_truth": multirc_ground_truth_str,
    "alexa": ["TODO"] * len(multirc_sample)
    }
)
    
print(multirc_temp)
glue_dfs.append(multirc_temp)


### parse qnli set ###
qnli_df = pl.read_ndjson(qnli_path)
qnli_sample = qnli_df.sample(2, with_replacement=False)

## Convert input question to string
qnli_input_question_str = qnli_sample \
    .select(pl.format("{question: {}, sentence: {}, Does the question entail the sentence?}",
                        pl.col("question"),
                        pl.col("sentence")))

## Convert ground_truth to string
qnli_ground_truth_str = qnli_sample.select(pl.format("{}",
                                                     pl.col("label")))

qnli_temp = pl.DataFrame(
    {
    "id": qnli_sample.select("idx"),
    "dataset": ["glue"] * len(qnli_sample),
    "subset": ["qnli"] * len(qnli_sample),
    "test": ["null"] * len(qnli_sample),
    "input_question": qnli_input_question_str,
    "ground_truth": qnli_ground_truth_str,
    "alexa": ["TODO"] * len(qnli_sample)
    }
)
    
print(qnli_temp)
glue_dfs.append(qnli_temp)


### parse qqp set ###
qqp_df = pl.read_ndjson(qqp_path)
qqp_sample = qqp_df.sample(2, with_replacement=False)

## Convert input question to string
qqp_input_question_str = qqp_sample \
    .select(pl.format("{question 1: {}, question 2: {}, Are these two questions the same?}",
                        pl.col("question1"),
                        pl.col("question2")))

## Convert ground_truth to string
qqp_ground_truth_str = qqp_sample \
    .select(pl.format("{}",
                        pl.col("label")))

qqp_temp = pl.DataFrame(
    {
    "id": qqp_sample.select("idx"),
    "dataset": ["glue"] * len(qqp_sample),
    "subset": ["qqp"] * len(qqp_sample),
    "test": ["null"] * len(qqp_sample),
    "input_question": qqp_input_question_str,
    "ground_truth": qqp_ground_truth_str,
    "alexa": ["TODO"] * len(qqp_sample)
    }
)
    
print(qqp_temp)
glue_dfs.append(qqp_temp)


### parse rte set ###
rte_df = pl.read_ndjson(rte_path)
rte_sample = rte_df.sample(2, with_replacement=False)
    
## Convert input question to string
rte_input_question_str = rte_sample \
    .select(pl.format("{sentence 1: {}, sentence 2: {}, 0 if there is entailment, 1 if not}",
                        pl.col("sentence1"),
                        pl.col("sentence2")))

## Convert ground_truth to string
rte_ground_truth_str = rte_sample.select(pl.format("{}",
                                                     pl.col("label")))

rte_temp = pl.DataFrame(
    {
    "id": rte_sample.select("idx"),
    "dataset": ["superglue"] * len(rte_sample),
    "subset": ["rte"] * len(rte_sample),
    "test": ["null"] * len(rte_sample),
    "input_question": rte_input_question_str,
    "ground_truth": rte_ground_truth_str,
    "alexa": ["TODO"] * len(rte_sample)
    }
)
    
print(rte_temp)
glue_dfs.append(rte_temp)


### parse sst2 set ###
sst2_df = pl.read_ndjson(sst2_path)
sst2_sample = sst2_df.sample(2, with_replacement=False)
    
## Convert input question to string
sst2_input_question_str = sst2_sample \
    .select(pl.format("{review: {}, Is this positive?}",
                        pl.col("sentence")))

## Convert ground_truth to string
sst2_ground_truth_str = sst2_sample.select(pl.format("{}",
                                                     pl.col("label")))

sst2_temp = pl.DataFrame(
    {
    "id": sst2_sample.select("idx"),
    "dataset": ["glue"] * len(sst2_sample),
    "subset": ["sst2"] * len(sst2_sample),
    "test": ["null"] * len(sst2_sample),
    "input_question": sst2_input_question_str,
    "ground_truth": sst2_ground_truth_str,
    "alexa": ["TODO"] * len(sst2_sample)
    }
)
    
print(sst2_temp)
glue_dfs.append(sst2_temp)

### parse wsc set ###
wsc_df = pl.read_ndjson(wsc_path)
wsc_sample = wsc_df.sample(2, with_replacement=False)
    
## Convert input question to string
wsc_input_question_str = wsc_sample \
    .select(pl.format("{text: {}, Does {} refer to {}?}",
                        pl.col("text"),
                        pl.col("span2_text"),
                        pl.col("span1_text")))

## Convert ground_truth to string
wsc_ground_truth_str = wsc_sample.select(pl.format("{}",
                                                     pl.col("label")))

wsc_temp = pl.DataFrame(
    {
    "id": wsc_sample.select("idx"),
    "dataset": ["superglue"] * len(wsc_sample),
    "subset": ["wsc"] * len(wsc_sample),
    "test": ["null"] * len(wsc_sample),
    "input_question": wsc_input_question_str,
    "ground_truth": wsc_ground_truth_str,
    "alexa": ["TODO"] * len(wsc_sample)
    }
)
    
print(wsc_temp)
glue_dfs.append(wsc_temp)



### export to csv ###

glue_df = pl.concat(glue_dfs)
print(glue_df)
path: pathlib.Path = "/Users/goose/Documents/CPSC 448/Evaluation Analysis/results/glue_filtered_analysis.csv"
glue_df.write_csv(path, separator=",")

#########

""" ### parse template set ###
template_df = pl.scan_csv(template_path, separator='\t', has_header=False, with_column_names=lambda cols: ["id", "label", "anno", "sentence"]).collect()
template_sample = template_df.sample(2, with_replacement=False)
    
## Convert ground_truth to string

ground_truth_str = template_sample \
    .select(pl.format("{}",
                        pl.col("label")))

template_temp = pl.DataFrame(
    {
    "id": template_sample.select("id"),
    "dataset": ["glue"] * len(template_sample),
    "subset": ["template"] * len(template_sample),
    "test": ["null"] * len(template_sample),
    "input_question": template_sample.select("sentence"),
    "ground_truth": ground_truth_str,
    "alexa": ["TODO"] * len(template_sample)
    }
)
    
print(template_temp)
glue_dfs.append(template_temp) """