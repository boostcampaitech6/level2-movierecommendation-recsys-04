0. directory 
- cd recbole

0-1. requirements
- pip install -r requirements.txt

0-2. make inter file for recbole
- python make_inter.py

1. train
- python main.py --model=[모델이름] --dataset=[학습할 데이터셋] --params_files=[파라미터 파일]

2. inferece
2-1. python run_inference.py
- 일반적인 모델의 경우 사용합니다.
- python run_inference.py --model_path=[pth파일의 경로]
    - 파일의 경로를 입력했을 때 오류가 나온다면 '\'를 '/'로 바꿔서 시도하십시오.
- pth파일이 저장된 saved파일에 pth파일의 이름 + submission.csv로 저장 

2-2. python run_inference_general_recommender.py
- 모델이 general_recommender의 경우 사용합니다.
- python run_inference_general_recommender.py --model_path=[pth파일의 경로]
    - 파일의 경로를 입력했을 때 오류가 나온다면 '\'를 '/'로 바꿔서 시도하십시오.
- pth파일이 저장된 saved파일에 pth파일의 이름 + submission.csv로 저장 
