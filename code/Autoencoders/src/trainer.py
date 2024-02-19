import time
import os
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import pandas as pd
import bottleneck as bn

from scipy import sparse
from .metric import NDCG_binary_at_k_batch, Recall_at_k_batch, Recall_at_k_batch
from .dataloader import filter_triplets


def sparse2torch_sparse(data):
    """
    Convert scipy sparse matrix to torch sparse tensor with L2 Normalization
    This is much faster than naive use of torch.FloatTensor(data.toarray())
    https://discuss.pytorch.org/t/sparse-tensor-use-cases/22047/2
    """
    samples = data.shape[0]
    features = data.shape[1]
    coo_data = data.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    row_norms_inv = 1 / np.sqrt(data.sum(1))
    row2val = {i: row_norms_inv[i].item() for i in range(samples)}
    values = np.array([row2val[r] for r in coo_data.row])
    t = torch.sparse.FloatTensor(
        indices, torch.from_numpy(values).float(), [samples, features]
    )
    return t


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


def train(config, model, train_data, criterion, optimizer, is_VAE=False):
    parameters = config["parameters"]

    N = train_data.shape[0]
    idxlist = list(range(N))

    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()
    update_count = 0

    np.random.shuffle(idxlist)

    for batch_idx, start_idx in enumerate(range(0, N, parameters["batch_size"])):
        end_idx = min(start_idx + parameters["batch_size"], N)
        data = train_data[idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(config["device"])
        optimizer.zero_grad()

        if is_VAE:
            if parameters["total_anneal_steps"] > 0:
                anneal = min(
                    parameters["anneal_cap"],
                    1.0 * update_count / parameters["total_anneal_steps"],
                )
            else:
                anneal = parameters["anneal_cap"]

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)

            loss = criterion(recon_batch, data, mu, logvar, anneal)
        else:
            recon_batch = model(data)
            loss = criterion(recon_batch, data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count += 1

        if batch_idx % parameters["log_interval"] == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | "
                "loss {:4.2f}".format(
                    parameters["epoch"],
                    batch_idx,
                    len(range(0, N, parameters["batch_size"])),
                    elapsed * 1000 / parameters["log_interval"],
                    train_loss / parameters["log_interval"],
                )
            )

            start_time = time.time()
            train_loss = 0.0


def evaluate(config, model, criterion, train_data, data_tr, data_te, is_VAE):
    parameters = config["parameters"]

    N = train_data.shape[0]

    # Turn on evaluation mode
    model.eval()
    update_count = 0
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    total_val_loss_list = []
    n100_list = []
    r20_list = []
    r50_list = []

    with torch.no_grad():
        for start_idx in range(0, e_N, parameters["batch_size"]):
            end_idx = min(start_idx + parameters["batch_size"], N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(config["device"])
            if is_VAE:

                if parameters["total_anneal_steps"] > 0:
                    anneal = min(
                        parameters["anneal_cap"],
                        1.0 * update_count / parameters["total_anneal_steps"],
                    )
                else:
                    anneal = parameters["anneal_cap"]

                recon_batch, mu, logvar = model(data_tensor)

                loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)

            else:
                recon_batch = model(data_tensor)
                loss = criterion(recon_batch, data_tensor)

            total_val_loss_list.append(loss.item())

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)

    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    return (
        np.nanmean(total_val_loss_list),
        np.nanmean(n100_list),
        np.nanmean(r20_list),
        np.nanmean(r50_list),
    )


def loss_function_vae(recon_x, x, mu, logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD


def loss_function_dae(recon_x, x):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    return BCE


def run(
    config, model, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te
):
    parameters = config["parameters"]
    best_n100 = -np.inf
    update_count = 0

    N = train_data.shape[0]

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=parameters["wd"])
    if config["model"] == "MultiDAE":
        criterion = loss_function_dae
        is_VAE = False
    elif config["model"] == "MultiVAE":
        criterion = loss_function_vae
        is_VAE = True

    for epoch in range(1, parameters["epochs"] + 1):
        epoch_start_time = time.time()
        train(config, model, train_data, criterion, optimizer, is_VAE)
        val_loss, n100, r20, r50 = evaluate(
            config, model, criterion, train_data, vad_data_tr, vad_data_te, is_VAE
        )
        print("-" * 89)
        print(
            "| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | "
            "n100 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}".format(
                epoch, time.time() - epoch_start_time, val_loss, n100, r20, r50
            )
        )
        print("-" * 89)

        n_iter = epoch * len(range(0, N, parameters["batch_size"]))

        # Save the model if the n100 is the best we've seen so far.
        if n100 > best_n100:
            with open(
                os.path.join(
                    config["model_save_path"],
                    f"{config['model']}_V_{config['config_ver']}.pt",
                ),
                "wb",
            ) as f:
                torch.save(model, f)
            best_n100 = n100

    # Load the best saved model.
    with open(
        os.path.join(
            config["model_save_path"], f"{config['model']}_V_{config['config_ver']}.pt"
        ),
        "rb",
    ) as f:
        model = torch.load(f)

    # Run on test data.
    test_loss, n100, r20, r50 = evaluate(
        config, model, criterion, train_data, test_data_tr, test_data_te, is_VAE
    )
    print("=" * 89)
    print(
        "| End of training | test loss {:4.2f} | n100 {:4.2f} | r20 {:4.2f} | "
        "r50 {:4.2f}".format(test_loss, n100, r20, r50)
    )
    print("=" * 89)


def inference(config, model, loader):
    raw_data = pd.read_csv(
        os.path.join(config["data_path"], "train_ratings.csv"), header=0
    )
    raw_data, user_activity, item_popularity = filter_triplets(
        raw_data, min_uc=5, min_sc=0
    )
    unique_uid = user_activity["user"].unique()

    n_users = unique_uid.size  # 31360
    n_heldout_users = 3000

    tr_users = unique_uid[: (n_users - n_heldout_users * 2)]
    train_plays = raw_data.loc[raw_data["user"].isin(tr_users)]
    unique_sid = pd.unique(train_plays["item"])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    # 전체 raw_data를 인덱싱하여 submit_data 데이터 프레임으로 생성
    submit_data = loader.load_data("submit")
    n_items = loader.load_n_items()

    n_users = submit_data["uid"].max() + 1
    rows, cols = submit_data["uid"], submit_data["sid"]

    # 시청 이력이 있으면 1, 없으면 0인 sparse매트릭스로 변환
    submit_csr_data = sparse.csr_matrix(
        (np.ones_like(rows), (rows, cols)), dtype="float64", shape=(n_users, n_items)
    )

    # inference
    model.eval()
    with torch.no_grad():
        data = submit_csr_data
        data_tensor = naive_sparse2tensor(data).to(config["device"])
        if config["model"] == "MultiDAE":
            recon_batch = model(data_tensor)
        if config["model"] == "MultiVAE":
            recon_batch, mu, logvar = model(data_tensor)
        recon_batch = recon_batch.cpu().numpy()
        recon_batch[data.nonzero()] = (
            -np.inf
        )  # 시청한 이력이 있으면 -np.inf로 더 이상 추천하지 않게 함
        idx = bn.argpartition(
            -recon_batch, 10, axis=1
        )  # 높은 점수 10개를 앞의 10개 위치로 이동

    # idx 10개 인덱스를 뽑아 최종 출력 데이터로 변환
    inference_df = (
        pd.DataFrame(idx[:, :10]).stack().reset_index().drop("level_1", axis=1)
    )
    inference_df.columns = ["user", "item"]

    # 인덱싱 원상복구 및 정렬
    inference_df["user"] = inference_df["user"].apply(lambda x: list(profile2id)[x])
    inference_df["item"] = inference_df["item"].apply(lambda x: list(show2id)[x])
    inference_df = inference_df.sort_values("user", ascending=True)

    # 데이터 export
    inference_df.to_csv(
        os.path.join(
            config["submit_path"], f"{config['model']}_V_{config['config_ver']}.csv"
        ),
        index=False,
    )

    print("Inference Done!")
