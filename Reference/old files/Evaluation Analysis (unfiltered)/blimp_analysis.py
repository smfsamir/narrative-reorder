import polars as pl
import pathlib
import os

blimp_dir = "/Users/goose/Documents/CPSC 448/Sets/BLiMP/iterable"
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
    good_sample = in_df.sample(2, with_replacement=False)
    bad_sample = in_df.sample(2, with_replacement=False)
    
    good_df = pl.DataFrame(
        {
            "id": good_sample.select("pairID"),
            "dataset": ["blimp"] * len(good_sample),
            "subset": good_sample.select("linguistics_term"),
            "test": good_sample.select("UID"),
            "input_question": good_sample.select("sentence_good"),
            "ground_truth": ["sentence_good"] * len(good_sample),
            "alexa": ["TODO"] * len(good_sample)
        }
    )

    bad_df = pl.DataFrame(
        {
            "id": bad_sample.select("pairID"),
            "dataset": ["blimp"] * len(bad_sample),
            "subset": bad_sample.select("linguistics_term"),
            "test": bad_sample.select("UID"),
            "input_question": bad_sample.select("sentence_bad"),
            "ground_truth": ["sentence_bad"] * len(bad_sample),
            "alexa": ["TODO"] * len(bad_sample)
        }
    )
    
    print(good_df)
    print(bad_df)
    blimp_dfs.append(good_df)
    blimp_dfs.append(bad_df)

blimp_df = pl.concat(blimp_dfs)
print(blimp_df)
path: pathlib.Path = "/Users/goose/Documents/CPSC 448/Evaluation Analysis/Results/blimp_analysis.csv"
blimp_df.write_csv(path, separator=",")