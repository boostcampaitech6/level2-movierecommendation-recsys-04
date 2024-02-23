# https://hoon-bari.github.io/RS/Recbole
# https://mingchin.tistory.com/420


import argparse
import torch
import numpy as np
import pandas as pd
import pickle
import os
from recbole.quick_start import load_data_and_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #feat siyun : train에서 진행했던 모델의 결과를 여기에 입력 
    parser.add_argument('--model_path', '-m', type=str, default='', help='name of models')
    
    
    args, _ = parser.parse_known_args()
    
    # model, dataset 불러오기
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(args.model_path)
    
    # device 설정
    device = config.final_config_dict['device']
    
    # user, item id -> token 변환 array
    user_id2token = dataset.field2id_token['user_id']
    item_id2token = dataset.field2id_token['item_id']
    
    # user-item sparse matrix
    matrix = dataset.inter_matrix(form='csr')

    # user id, predict item id 저장 변수
    pred_list = None
    user_list = None
    
    model.eval()
    for data in test_data:
        interaction = data[0].to(device)
        
        score = model.full_sort_predict(interaction)
        
        rating_pred = score.cpu().data.numpy().copy()
        batch_user_index = interaction['user_id'].cpu().numpy()
        rating_pred[matrix[batch_user_index].toarray() > 0] = 0
        ind = np.argpartition(rating_pred, -10)[:, -10:]
        
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        batch_pred_list = ind[
            np.arange(len(rating_pred))[:, None], arr_ind_argsort
        ]
        
        # 예측값 저장
        if pred_list is None:
            pred_list = batch_pred_list
            user_list = batch_user_index
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            user_list = np.append(user_list, batch_user_index, axis=0)
        
    result = []
    for user, pred in zip(user_list, pred_list):
        for item in pred:
            result.append((int(user_id2token[user]), int(item_id2token[item])))
            
    # 데이터 저장
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    result_name = f"{model_name}_submission.csv"
    dataframe = pd.DataFrame(result, columns=["user", "item"])
    dataframe.sort_values(by='user', inplace=True)
    # 저장될 파일의 경로 지정
    ## ex) /data/ephemeral/RECBOLE/RecBole/saved/EASE_hyper_submission.csv
    dataframe.to_csv(
        os.path.join("saved/",result_name) ,index=False)

    
    print('inference done!')
    
    

        