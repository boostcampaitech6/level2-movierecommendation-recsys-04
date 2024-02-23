import os
import pandas as pd

'''
# 기존 dataset ml-1m을 바탕으로 칼럼을 만들어줘야 recbole을 사용할 수 있음.

ml-1m.inter
|---------------|---------------|--------------|-----------------|
| user_id:token | item_id:token | rating:float | timestamp:float |
|---------------|---------------|--------------|-----------------|
| 1             | 1193          | 5            | 978300760       |
| 1             | 661           | 3            | 978302109       |
|---------------|---------------|--------------|-----------------|

ml-1m.user
|---------------|-----------|--------------|------------------|----------------|
| user_id:token | age:token | gender:token | occupation:token | zip_code:token |
|---------------|-----------|--------------|------------------|----------------|
| 1             | 1193      | F            | 10               | 48067          |
| 2             | 661       | M            | 16               | 70072          |
|---------------|-----------|--------------|------------------|----------------|

ml-1m.item
|---------------|-----------------------|--------------------|-------------------|
| item_id:token | movie_title:token_seq | release_year:token | genre:token_seq   |
|---------------|-----------------------|--------------------|-------------------|
| 1             | Toy Story             | 5                  | Animation Comedy  |
| 2             | Jumanji               | 3                  | Adventure         |
|---------------|-----------------------|--------------------|-------------------|
'''

# recbole의 dataset 파일에 올리기 위해 파일 경로 지정 및 폴더 생성
## target_dir에 inter파일이 생성됨
TARGET_DIR = os.path.join(os.getcwd(), "dataset/movie")
os.makedirs(TARGET_DIR, exist_ok=True)
print("Creating New Data for RecBole...")


# === movie.inter 생성
# train_ratings.csv를 받아서 movie.inter로 생성
FILE = os.path.join(os.getcwd(), 'data/train/train_ratings.csv')
TARGET_NAME = 'movie.inter'


df = pd.read_csv(FILE)
df = df.rename(
    columns = {
        'user': 'user_id:token',
        'item': 'item_id:token',
        'time': 'timestamp:float',
    }
)
df.to_csv(os.path.join(TARGET_DIR, TARGET_NAME), index=False, sep='\t')
print(f'✅ {FILE} Created {TARGET_NAME}!')


# === unique_user.csv
# train_ratings.csv를 받아서 유저의 unique만 따로 movie.user로 생성
FILE = os.path.join(os.getcwd(), 'data/train/train_ratings.csv')
TARGET_NAME = 'unique_user.csv'

df = pd.read_csv(FILE)
df = pd.DataFrame(df['user'].unique(), columns=['user'])
df.to_csv(os.path.join(TARGET_DIR, TARGET_NAME), index=False)
print(f'✅ Created {TARGET_NAME}!')


# === movie.item
# === movie 정보들 merge 한 movie.csv 파일 만들기
# 데이터 불러오기

data_path = os.path.join(os.getcwd(), 'data/train/')
train = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
directors = pd.read_csv(os.path.join(data_path, 'directors.tsv'), sep='\t')
genres = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep='\t')
titles = pd.read_csv(os.path.join(data_path, 'titles.tsv'), sep='\t')
writers = pd.read_csv(os.path.join(data_path, 'writers.tsv'), sep='\t')
years = pd.read_csv(os.path.join(data_path, 'years.tsv'), sep='\t')


# 각 아이템 별 feature 정보 불러오기
# 하나의 아이템이 여러 장르 등에 속할 수 있음
def token_seq(feat):
    feat = list(set(feat))
    return " ".join(feat)


def process_data(df, column):
    merged_df = pd.merge(train, df, on=['item'])
    merged_df = merged_df.drop(['user', 'time'], axis=1)
    new_data = merged_df.groupby('item')[column].apply(token_seq)
    return new_data


new_genres = process_data(genres, 'genre')
new_writers = process_data(writers, 'writer')
new_directors = process_data(directors, 'director')


# merge
df = pd.merge(years, new_genres, on=['item'])   # year 정보는 item 당 한가지 값만 가짐
df = pd.merge(df, new_writers, on=['item'])
df = pd.merge(df, new_directors, on=['item'])

# recbole의 movie파일에 생성
# df.to_csv('/data/ephemeral/RECBOLE/RecBole/dataset/movie/movie.csv', index=False)

df.to_csv(os.path.join(os.getcwd(), "/dataset/movie/movie.csv"), index=False)

# === movie.csv 에서 unique_user.csv 만들기
FILE = os.path.join(os.getcwd(), "/dataset/movie/movie.csv")
TARGET_NAME = 'unique_user.csv'

df = pd.read_csv(FILE)
df = df.rename(
    columns={
        'item': 'item_id:token',
        'year': 'year:token',
        'genre': 'genre:token_seq',
        'writer': 'writer:token_seq',
        'director': 'director:token_seq',
    }
)

df.to_csv(os.path.join(TARGET_DIR, TARGET_NAME), index=False, sep='\t')
print(f'✅ Created {TARGET_NAME}!')

print('Done! Successfully Created New Dataset for RecBole')
