import os
import pandas as pd
from scipy import sparse
import numpy as np


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()

    return count


# 특정한 횟수 이상의 리뷰가 존재하는(사용자의 경우 min_uc 이상, 아이템의 경우 min_sc이상)
# 데이터만을 추출할 때 사용하는 함수입니다.
# 현재 데이터셋에서는 결과적으로 원본그대로 사용하게 됩니다.
def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, "item")
        tp = tp[tp["item"].isin(itemcount[itemcount["size"] >= min_sc]["item"])]

    if min_uc > 0:
        usercount = get_count(tp, "user")
        tp = tp[tp["user"].isin(usercount[usercount["size"] >= min_uc]["user"])]

    usercount, itemcount = get_count(tp, "user"), get_count(tp, "item")

    print("유저별 리뷰수: ", usercount)
    print("아이템별 리뷰수: ", itemcount)

    return tp, usercount, itemcount


# 훈련된 모델을 이용해 검증할 데이터를 분리하는 함수입니다.
# 100개의 액션이 있다면, 그중에 test_prop 비율 만큼을 비워두고, 그것을 모델이 예측할 수 있는지를
# 확인하기 위함입니다.
def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby("user")
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype="bool")
            idx[
                np.random.choice(
                    n_items_u, size=int(test_prop * n_items_u), replace=False
                ).astype("int64")
            ] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        else:
            tr_list.append(group)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    uid = tp["user"].apply(lambda x: profile2id[x])
    sid = tp["item"].apply(lambda x: show2id[x])
    return pd.DataFrame(data={"uid": uid, "sid": sid}, columns=["uid", "sid"])


def DataPreprocess(data_path):
    raw_data = pd.read_csv(os.path.join(data_path, "train_ratings.csv"), header=0)
    raw_data, user_activity, item_popularity = filter_triplets(
        raw_data, min_uc=5, min_sc=0
    )


class DataLoader:
    """
    Load Movielens dataset
    """

    def __init__(self, path):

        self.pro_dir = os.path.join(path, "pro_sg")
        assert os.path.exists(
            self.pro_dir
        ), "Preprocessed files do not exist. Run data.py"

        self.n_items = self.load_n_items()

    def load_data(self, datatype="train"):
        if datatype == "train":
            return self._load_train_data()
        elif datatype == "validation":
            return self._load_tr_te_data(datatype)
        elif datatype == "test":
            return self._load_tr_te_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")

    def load_n_items(self):
        unique_sid = list()
        with open(os.path.join(self.pro_dir, "unique_sid.txt"), "r") as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items

    def _load_train_data(self):
        path = os.path.join(self.pro_dir, "train.csv")

        tp = pd.read_csv(path)
        n_users = tp["uid"].max() + 1

        rows, cols = tp["uid"], tp["sid"]
        data = sparse.csr_matrix(
            (np.ones_like(rows), (rows, cols)),
            dtype="float64",
            shape=(n_users, self.n_items),
        )
        return data

    def _load_tr_te_data(self, datatype="test"):
        tr_path = os.path.join(self.pro_dir, "{}_tr.csv".format(datatype))
        te_path = os.path.join(self.pro_dir, "{}_te.csv".format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        start_idx = min(tp_tr["uid"].min(), tp_te["uid"].min())
        end_idx = max(tp_tr["uid"].max(), tp_te["uid"].max())

        rows_tr, cols_tr = tp_tr["uid"] - start_idx, tp_tr["sid"]
        rows_te, cols_te = tp_te["uid"] - start_idx, tp_te["sid"]

        data_tr = sparse.csr_matrix(
            (np.ones_like(rows_tr), (rows_tr, cols_tr)),
            dtype="float64",
            shape=(end_idx - start_idx + 1, self.n_items),
        )
        data_te = sparse.csr_matrix(
            (np.ones_like(rows_te), (rows_te, cols_te)),
            dtype="float64",
            shape=(end_idx - start_idx + 1, self.n_items),
        )
        return data_tr, data_te
