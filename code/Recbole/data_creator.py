import os

import pandas as pd

# SETTINGS
TARGET_DIR = os.path.join(os.getcwd(), "data/movie")
os.makedirs(TARGET_DIR, exist_ok=True)
print("Data Creation Start!")


# make unique_user.csv
FILE = "/data/ephemeral/data/train/train_ratings.csv"
TARGET_NAME = "unique_user.csv"

df = pd.read_csv(FILE)
unique = df["user"].unique()
df = pd.DataFrame(unique, columns=["user"])
df.to_csv(os.path.join(TARGET_DIR, TARGET_NAME), index=False)
print("Create unique_user.csv!")


# make movie.inter
FILE = "/data/ephemeral/data/train/train_ratings.csv"
TARGET_NAME = "movie.inter"

df = pd.read_csv(FILE)
df["label"] = 1
df = df.rename(
    columns={
        "user": "user_id:token",
        "item": "item_id:token",
        "time": "timestamp:float",
        "label": "label:float",
    }
)
df.to_csv(os.path.join(TARGET_DIR, TARGET_NAME), index=False, sep="\t")
print("Create movie.inter!")

data_path = "/data/ephemeral/data/train/"
train = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
directors = pd.read_csv(os.path.join(data_path, "directors.tsv"), sep="\t")
genres = pd.read_csv(os.path.join(data_path, "genres.tsv"), sep="\t")
titles = pd.read_csv(os.path.join(data_path, "titles.tsv"), sep="\t")
writers = pd.read_csv(os.path.join(data_path, "writers.tsv"), sep="\t")
years = pd.read_csv(os.path.join(data_path, "years.tsv"), sep="\t")

titles["title"] = titles[titles["title"].astype(str).str[-6] == "("]["title"].apply(
    lambda x: x[:-7]
)

writers = writers.sort_values(["item", "writer"])
writers = writers.groupby("item")["writer"].agg(" ".join).reset_index()

genres = genres.sort_values(["item", "genre"])
genres = genres.groupby("item")["genre"].agg(" ".join).reset_index()

directors = directors.sort_values(["item", "director"])
directors = directors.groupby("item")["director"].agg(" ".join).reset_index()

item_merge = titles.merge(years, how="outer", on="item")
item_merge = item_merge.merge(writers, how="outer", on="item")
item_merge = item_merge.merge(genres, how="outer", on="item")
item_merge = item_merge.merge(directors, how="outer", on="item")

item_merge.to_csv("data/movie/movie.csv", index=False)
print("Create movie.csv!")


# make movie.item
FILE = "data/movie/movie.csv"
TARGET_NAME = "movie.item"

df = pd.read_csv(FILE)
df = df.rename(
    columns={
        "item": "item_id:token",
        "title": "title:token_seq",
        "year": "year:token",
        "genre": "genre:token_seq",
        "writer": "writer:token_seq",
        "director": "director:token_seq",
    }
)
df.to_csv(os.path.join(TARGET_DIR, TARGET_NAME), index=False, sep="\t")
print("Create movie.item!")
