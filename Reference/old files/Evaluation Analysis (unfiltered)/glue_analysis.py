import polars as pl
import pathlib
import os

cola_path = "/Users/goose/Documents/CPSC 448/Sets/GLUE/CoLA/train.tsv"
mnli_path = "/Users/goose/Documents/CPSC 448/Sets/GLUE/MNLI/train.tsv"
qnli_path = "/Users/goose/Documents/CPSC 448/Sets/GLUE/QNLI/train.tsv"
qqp_path = "/Users/goose/Documents/CPSC 448/Sets/GLUE/QQP/train.tsv"
rte_path = "/Users/goose/Documents/CPSC 448/Sets/GLUE/RTE/train.tsv"
sst2_path = "/Users/goose/Documents/CPSC 448/Sets/GLUE/SST-2/train.tsv"

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

### parse cola set ###
cola_df = pl.scan_csv(cola_path, separator='\t', has_header=False, with_column_names=lambda cols: ["id", "label", "anno", "sentence"]).collect()
cola_sample = cola_df.sample(2, with_replacement=False)
    
## Convert ground_truth to string
cola_ground_truth_str = cola_sample \
    .select(pl.format("{}",
                        pl.col("label")))

cola_temp = pl.DataFrame(
    {
    "id": cola_sample.select("id"),
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
mnli_df = pl.scan_csv(mnli_path, separator='\t', truncate_ragged_lines=True).collect()
mnli_sample = mnli_df.sample(2, with_replacement=False)

## Convert input question to string
mnli_input_question_str = mnli_sample \
    .select(pl.format("{{}, {}, Is the second sentence entailment, contradiction, or neutral to the first?}",
                        pl.col("sentence1"),
                        pl.col("sentence2")))

mnli_temp = pl.DataFrame(
    {
    "id": mnli_sample.select("pairID"),
    "dataset": ["glue"] * len(mnli_sample),
    "subset": ["mnli"] * len(mnli_sample),
    "test": ["null"] * len(mnli_sample),
    "input_question": mnli_input_question_str,
    "ground_truth": mnli_sample.select("gold_label"),
    "alexa": ["TODO"] * len(mnli_sample)
    }
)
    
print(mnli_temp)
glue_dfs.append(mnli_temp)


### parse qnli set ###
qnli_df = pl.scan_csv(qnli_path, separator='\t', truncate_ragged_lines=True).collect()
qnli_sample = qnli_df.sample(2, with_replacement=False)

## Convert id to string
qnli_id_str = qnli_sample \
    .select(pl.format("{}",
                        pl.col("index")))

## Convert input question to string
qnli_input_question_str = qnli_sample \
    .select(pl.format("{{}, {}, Is the second sentence entailment to the first?}",
                        pl.col("question"),
                        pl.col("sentence")))

qnli_temp = pl.DataFrame(
    {
    "id": qnli_id_str,
    "dataset": ["glue"] * len(qnli_sample),
    "subset": ["qnli"] * len(qnli_sample),
    "test": ["null"] * len(qnli_sample),
    "input_question": qnli_input_question_str,
    "ground_truth": qnli_sample.select("label"),
    "alexa": ["TODO"] * len(qnli_sample)
    }
)
    
print(qnli_temp)
glue_dfs.append(qnli_temp)


### parse qqp set ###
qqp_df = pl.scan_csv(qqp_path, separator='\t').collect()
qqp_sample = qqp_df.sample(2, with_replacement=False)

## Convert id to string
qqp_id_str = qqp_sample \
    .select(pl.format("{}",
                        pl.col("id")))

## Convert input question to string
qqp_input_question_str = qqp_sample \
    .select(pl.format("{{}, {}, Are these two questions duplicates?}",
                        pl.col("question1"),
                        pl.col("question2")))

## Convert ground_truth to string
qqp_ground_truth_str = qqp_sample \
    .select(pl.format("{}",
                        pl.col("is_duplicate")))

qqp_temp = pl.DataFrame(
    {
    "id": qqp_id_str,
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


### export to csv ###

glue_df = pl.concat(glue_dfs)
print(glue_df)
path: pathlib.Path = "/Users/goose/Documents/CPSC 448/Evaluation Analysis/Results/glue_analysis.csv"
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