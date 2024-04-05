# NEEDS TO BE REDONE; TODO

import polars as pl
import pathlib
import os

msgs_filtered_dir = "/Users/goose/Documents/CPSC 448/Evaluation Analysis/data/msgs_filtered/train"

msgs_dfs = []

for file in os.scandir(msgs_filtered_dir):

    # grab file path
    f = file.path
    print(f)

    in_df = pl.read_ndjson(f)
    sampled_df = in_df.sample(2, with_replacement=False)

    input_question_str = ""
    ground_truth_str = ""

    if sampled_df.filter((pl.col("linguistic_feature_description") != "null") & 
                         (pl.col("surface_feature_description") != "null")).shape[0] > 0:

        # collect mixed dataset

        # Convert struct columns to formatted string
        input_question_str = sampled_df.select(pl.format("{}, {}, {}",
                                                        pl.col("sentence"),
                                                        pl.col("linguistic_feature_description"),
                                                        pl.col("surface_feature_description")))

        ground_truth_str = sampled_df.select(pl.format("{}, {}",
                                                       pl.col("linguistic_feature_label"),
                                                       pl.col("surface_feature_label")))

    elif sampled_df.filter(pl.col("surface_feature_description") == "null").shape[0] > 0:

        # collect linguistics datasets

        # Convert struct columns to formatted string
        input_question_str = sampled_df.select(pl.format("{{}}, {{}}",
                                                        pl.col("sentence"),
                                                        pl.col("linguistic_feature_description")))

        ground_truth_str = sampled_df.select(pl.format("{{}}",
                                                       pl.col("linguistic_feature_label")))

    elif sampled_df.filter(pl.col("linguistic_feature_description") == "null").shape[0] > 0:
        # collect surface datasets

        # Convert struct columns to formatted string
        input_question_str = sampled_df.select(pl.format("{}, {}",
                                                        pl.col("sentence"),
                                                        pl.col("surface_feature_description")))

        ground_truth_str = sampled_df.select(pl.format("{}",
                                                       pl.col("surface_feature_label")))

    temp_df = pl.DataFrame(
        {
            "id": sampled_df.select("sentenceID"),
            "dataset": ["msgs"] * len(sampled_df),
            "subset": ["null"] * len(sampled_df),
            "test": sampled_df.select("UID"),
            "input_question": input_question_str,
            "ground_truth": ground_truth_str,
            "alexa": ["TODO"] * len(sampled_df)
        }
    )

    print(temp_df)
    msgs_dfs.append(temp_df)

msgs_df = pl.concat(msgs_dfs)
print(msgs_df)
path: pathlib.Path = "/Users/goose/Documents/CPSC 448/Evaluation Analysis/results/msgs_filtered_analysis.csv"
msgs_df.write_csv(path, separator=",")