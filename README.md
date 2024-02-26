![Untitled](https://github.com/boostcampaitech6/level2-movierecommendation-recsys-04/assets/8871767/2f6f5686-3313-47d2-aa04-261a4896e782)

# 목차
### [Team](#Team-1)
### [Skill](#Skill-1)
### [Project Overview](#Project-Overview-1)
### [Project Structure](#Project-Structure-1)
&nbsp;&nbsp;[Calendar](#Calendar)<br>
&nbsp;&nbsp;[Pipeline](#Pipeline)<br>
&nbsp;&nbsp;[1. Environment](#1-Environment)<br>
&nbsp;&nbsp;[2. Data](#2-Data)<br>
&nbsp;&nbsp;[3. Model](#3-Model)<br>
&nbsp;&nbsp;[4. Performance](#5-Performance)<br> 
### [Laboratory Report](#Laboratory-Report-1)

# Team
| **김세훈** | **문찬우** | **김시윤** | **배건우** | **이승준** |
| :------: |  :------: | :------: | :------: | :------: |
| [<img src="https://avatars.githubusercontent.com/u/8871767?v=4" height=150 width=150>](https://github.com/warpfence) | [<img src="https://avatars.githubusercontent.com/u/95879995?v=4" height=150 width=150> ](https://github.com/chanwoomoon) | [<img src="https://avatars.githubusercontent.com/u/68991530?v=4" height=150 width=150> ](https://github.com/tldbs5026) | [<img src="https://avatars.githubusercontent.com/u/83867930?v=4" height=150 width=150>](https://github.com/gunwoof) | [<img src="https://avatars.githubusercontent.com/u/133944361?v=4" height=150 width=150>](https://github.com/llseungjun) |
- 공통 : EDA, Hyper Parameter Tuning, Git Management, Recbole
- 김세훈 : MultiDAE, MultiVAE Baseline 구축 및 실험, Ensemble
- 문찬우 : 모델 성능 확인, 모델간의 유사도 확인 및 Ensemble
- 김시윤 : RecBole 실험환경 세팅(기본 환경, inference), ease, lightgcn, recvae, deepfm 모델 실험, 앙상블 진행
- 배건우 : 서버환경 구축, 베이스라인 구축
- 이승준 : SASRec Baseline 구축 및 실험, RecBole을 활용한 EASE, ADMM-SLIM, CDAE, GRU4Rec 모델 실험

# Skill 
### Language
  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

### Library
  ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
  ![scikitlearn](https://img.shields.io/badge/scikitlearn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
  ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
  ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
  ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ff0000.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

### Communication
  ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
  ![Github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)
  ![Wandb](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)
  ![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)
  ![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)

### Environment
  ![NVIDIA-TeslaV100](https://img.shields.io/badge/NVIDIA-TeslaV100-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
  ![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
  ![Anaconda](https://img.shields.io/badge/Anaconda-44A833.svg?style=for-the-badge&logo=Anaconda&logoColor=white)

# Project Overview

본 프로젝트는 사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 인공지능을 통해 예측하는 프로젝트이다.
 
![Untitled 1](https://github.com/boostcampaitech6/level2-movierecommendation-recsys-04/assets/8871767/800c4636-6f07-4522-9699-964aaf813af9)

# Project Structure

### Calendar
![Untitled 2](https://github.com/boostcampaitech6/level2-movierecommendation-recsys-04/assets/8871767/ba9ac29f-f1a2-4ff1-a4bf-d57a08f93a9c)

### Project Pipline
![Untitled 3](https://github.com/boostcampaitech6/level2-movierecommendation-recsys-04/assets/8871767/b85a8378-883d-409f-9c57-11f362ac6329)
### 1. Environment
```
numpy==1.22.2
pandas==1.4.1
python-dateutil==2.8.2
pytz==2021.3
recbole==1.2.0
scipy==1.8.0
six==1.16.0
torch==1.10.2
tqdm==4.62.3
typing_extensions==4.1.1
```
### 2. Data
**`userID`** : 사용자 별 고유번사로 총 7,422명의 사용자 데이터가 존재합니다.

**`assessmentItemID` :** 문항의 고유번호이며, 총 9,454개의 고유 문항이 있습니다.

**`testID` :** 시험지의 고유번호이며, 총 1,537개의 고유한 시험지가 있습니다.

**`answerCode` :** 사용자가 해당 문항을 맞췄는지 여부이며,  0은 틀릿 것, 1은 맞춘 것입니다. test 데이터의 경우 마지막 시퀀스의 answerCode가 -1로 예측해야 할 값입니다.

**`Timestamp` :** 사용자가 해당문항을 풀기 시작한 시점의 데이터입니다.

**`KnowleadgeTag` :** 문항 당 하나씩 배정되는 태그로, 일종의 중분류 역할을 합니다. 912개의 고유 태그가 존재합니다.

![feature](https://github.com/boostcampaitech6/level2-dkt-recsys-04/assets/68991530/198f77f4-ee69-4172-9033-0602a47cf6ba)
### 3. Model
  - **Boosting model**
    ![Boosting_Flow_Chart](https://github.com/boostcampaitech6/level2-dkt-recsys-04/assets/8871767/4031ba71-8ec2-4232-ab36-8fbc3e55f7bc)
  - **Sequence model**
    ![model_seq](https://github.com/boostcampaitech6/level2-dkt-recsys-04/assets/95879995/82b5668c-2b82-4038-8900-0ab418a64bad)

### 4. Performance

| **Model** | **LGBM-v1** | **Saint** | **Last-Query + GRU** | **LSTMATTN** | **GRUATTN** | **LGBM-v2** |
| :------: |  :------: | :------: | :------: | :------: | :------: | :------: |
| **Weight** | **0.67** | **0.084** | **0.064** | **0.064** | **0.059** | **0.059** | 

| **Public AUC** | **Public ACC** |
| :------: |  :------: | 
| 0.8156 | 0.7527 | 

![result](https://github.com/boostcampaitech6/level2-dkt-recsys-04/assets/68991530/ad3ecb4d-ce3d-4735-836e-318d1c998502)

# Laboratory Report
[DKT_Recsys_팀_리포트](https://github.com/boostcampaitech6/level2-dkt-recsys-04/blob/main/DKT_Recsys_%ED%8C%80_%EB%A6%AC%ED%8F%AC%ED%8A%B8(04%EC%A1%B0).pdf)


