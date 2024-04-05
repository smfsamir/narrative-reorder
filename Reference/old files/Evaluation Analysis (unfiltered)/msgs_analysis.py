import polars as pl
import pathlib
import os

msgs_mixed_dir = "/Users/goose/Documents/CPSC 448/Sets/MSGS/iterable/mixed"
msgs_ling_dir = "/Users/goose/Documents/CPSC 448/Sets/MSGS/iterable/ling"
msgs_surf_dir = "/Users/goose/Documents/CPSC 448/Sets/MSGS/iterable/surf"

msgs_dfs = []

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

# iterate over mixed datasets
for file in os.scandir(msgs_mixed_dir) :

    # grab file path
    f = file.path
    print(f)

    in_df = pl.read_ndjson(f)
    sampled_df = in_df.sample(2, with_replacement=False)
    
    # Convert struct columns to formatted string
    input_question_str = sampled_df \
        .select(pl.format("{{}, {}, {}}",
                          pl.col("sentence"),
                          pl.col("linguistic_feature_description"),
                          pl.col("surface_feature_description")))

    ground_truth_str = sampled_df \
        .select(pl.format("{{}, {}}",
                          pl.col("linguistic_feature_label"),
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

# iterate over linguistics datasets
for file in os.scandir(msgs_ling_dir) :

    # grab file path
    f = file.path
    print(f)

    in_df = pl.read_ndjson(f)
    sampled_df = in_df.sample(2, with_replacement=False)
    
    # Convert struct columns to formatted string
    input_question_str = sampled_df \
        .select(pl.format("{{}, {}}",
                          pl.col("sentence"),
                          pl.col("linguistic_feature_description")))

    ground_truth_str = sampled_df \
        .select(pl.format("{{}}",
                          pl.col("linguistic_feature_label")))

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

# iterate over surface datasets
for file in os.scandir(msgs_surf_dir) :

    # grab file path
    f = file.path
    print(f)

    in_df = pl.read_ndjson(f)
    sampled_df = in_df.sample(2, with_replacement=False)
    
    # Convert struct columns to formatted string
    input_question_str = sampled_df \
        .select(pl.format("{{}, {}}",
                          pl.col("sentence"),
                          pl.col("surface_feature_description")))
    
    ground_truth_str = sampled_df \
        .select(pl.format("{{}}",
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
path: pathlib.Path = "/Users/goose/Documents/CPSC 448/Evaluation Analysis/Filtered Results/msgs_filtered_analysis.csv"
msgs_df.write_csv(path, separator=",")