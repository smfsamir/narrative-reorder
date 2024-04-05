import polars as pl
import numpy as np
import pathlib
import os

blimp_dir = "/Users/goose/Documents/CPSC 448/Evaluation Analysis/data/blimp_filtered"
blimp_dfs = []

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

# iterate over all blimp datasets
for file in os.scandir(blimp_dir) :

    # grab file path
    f = file.path
    print(f)

    in_df = pl.read_ndjson(f)
    good_sample = in_df.sample(np.random.randint(1, 3), with_replacement=False)
    bad_sample = in_df.sample(np.random.randint(1, 3), with_replacement=False)
    
    good_question_str = good_sample.select(pl.format("{}, {}",
                                                        pl.col("sentence_good"),
                                                        pl.col("sentence_bad")))
    
    bad_question_str = bad_sample.select(pl.format("{}, {}",
                                                        pl.col("sentence_bad"),
                                                        pl.col("sentence_good")))

    good_df = pl.DataFrame(
        {
            "id": good_sample.select("pair_id"),
            "dataset": ["blimp"] * len(good_sample),
            "subset": good_sample.select("linguistics_term"),
            "test": good_sample.select("UID"),
            "input_question": good_question_str,
            "ground_truth": ["sentence_good, sentence_bad"] * len(good_sample),
            "alexa": ["TODO"] * len(good_sample)
        }
    )

    bad_df = pl.DataFrame(
        {
            "id": bad_sample.select("pair_id"),
            "dataset": ["blimp"] * len(bad_sample),
            "subset": bad_sample.select("linguistics_term"),
            "test": bad_sample.select("UID"),
            "input_question": bad_question_str,
            "ground_truth": ["sentence_bad, sentence_good"] * len(bad_sample),
            "alexa": ["TODO"] * len(bad_sample)
        }
    )
    
    print(good_df)
    print(bad_df)
    blimp_dfs.append(good_df)
    blimp_dfs.append(bad_df)

blimp_df = pl.concat(blimp_dfs)
print(blimp_df)
path: pathlib.Path = "/Users/goose/Documents/CPSC 448/Evaluation Analysis/results/blimp_analysis.csv"
blimp_df.write_csv(path, separator=",")