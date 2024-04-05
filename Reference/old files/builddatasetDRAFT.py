from pathlib import Path
from nltk.tokenize import sent_tokenize
import polars as pl 
import os

""" df_schema = pl.DataFrame(
    {   
        "sentence_id": pl.Int32,
        "paragraph_id": pl.Int32,
        "story_id": pl.Int32,
        "sentence": pl.String,
        "paragraph_index": pl.Int32,
        "sentence_index": pl.Int32,
        "title": pl.String,
        "source": pl.String,
        "URL": pl.String
    }
) """

dir = "/Users/goose/Documents/CPSC 448/Sentence Shuffling/data for use"
sources = [{'name': 'ChiSCor', 'path': "/Users/goose/Documents/CPSC 448/Sentence Shuffling/data for use/ChiSCor english text", 'URL': "https://osf.io/5h7za/?view_only=705c553e922046058ef9df52b4ac8ed7"},
           {'name': 'The Aesop for Children', 'path': "/Users/goose/Documents/CPSC 448/Sentence Shuffling/data for use/The Aesop for Children", 'URL': "https://www.gutenberg.org/cache/epub/19994/pg19994-images.html"}]

dfs = []
stories = []
stories_tokenized = []
paragraphs = []

sentenceid = 0
paragraphid = 0
storyid = 0


# iterate over data
for source in sources :
    for file in os.scandir(source['path']) :
        # grab file path
        f = file.path
        title = Path(f).stem

        if f.lower().endswith('.txt'):
            # Try different encodings until successful or no more to try
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    story = Path(f).read_text(encoding=encoding)
                    stories.append(story)
                    break  # If successful, exit the loop
                except UnicodeDecodeError:
                    print(f"Failed to decode using {encoding} encoding")

    for story in stories :
        p = story.split(2*os.linesep)
        stories_tokenized.append(p)

    # split paragraphs into their sentences
    for story in stories_tokenized : 
        parindex = 0
        for p in story :
            sentences = sent_tokenize(p)
            sentindex = 0
            for sentence in sentences:
                df = pl.DataFrame(
                    {
                        "sentence_id": sentenceid,
                        "paragraph_id": paragraphid,
                        "story_id": storyid,
                        "sentence": sentence,
                        "paragraph_index": parindex,
                        "sentence_index": sentindex,
                        "title": title,
                        "source": source['name'],
                        "URL": source['URL']
                    }
                )
                dfs.append(df)
                sentenceid += 1
                sentindex += 1
            parindex += 1
            paragraphid += 1
        storyid += 1

print(dfs)

aesop_df = pl.concat(dfs)
aesop_df.write_ndjson("/Users/goose/Documents/CPSC 448/Sentence Shuffling/sentencedataset.json")