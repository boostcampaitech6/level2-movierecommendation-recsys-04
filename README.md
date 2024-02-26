![Untitled](https://github.com/boostcampaitech6/level2-movierecommendation-recsys-04/assets/8871767/2f6f5686-3313-47d2-aa04-261a4896e782)
## Recsys-04 파이팅해야조
| **김세훈** | **문찬우** | **김시윤** | **배건우** | **이승준** |
| :------: |  :------: | :------: | :------: | :------: |
| [<img src="https://avatars.githubusercontent.com/u/8871767?v=4" height=150 width=150>](https://github.com/warpfence) | [<img src="https://avatars.githubusercontent.com/u/95879995?v=4" height=150 width=150> ](https://github.com/chanwoomoon) | [<img src="https://avatars.githubusercontent.com/u/68991530?v=4" height=150 width=150> ](https://github.com/tldbs5026) | [<img src="https://avatars.githubusercontent.com/u/83867930?v=4" height=150 width=150>](https://github.com/gunwoof) | [<img src="https://avatars.githubusercontent.com/u/133944361?v=4" height=150 width=150>](https://github.com/llseungjun) |
- 공통 : EDA, Hyper Parameter Tuning, Git Management, Recbole
- 김세훈 : MultiDAE, MultiVAE Baseline 구축 및 실험, Ensemble
- 문찬우 : 모델 성능 확인, 모델간의 유사도 확인 및 Ensemble
- 김시윤 : RecBole 실험환경 세팅(기본 환경, inference), ease, lightgcn, recvae, deepfm 모델 실험, 앙상블 진행
- 배건우 : 서버환경 구축, 베이스라인 구축
- 이승준 : SASRec Baseline 구축 및 실험, RecBole을 활용한 EASE, ADMM-SLIM, CDAE, GRU4Rec 모델 실험

# Project Overview
본 프로젝트는 사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 인공지능을 통해 예측하는 프로젝트이다.
![Untitled 1](https://github.com/boostcampaitech6/level2-movierecommendation-recsys-04/assets/8871767/800c4636-6f07-4522-9699-964aaf813af9)

# Project Structure
### 1. Calendar
![Untitled 2](https://github.com/boostcampaitech6/level2-movierecommendation-recsys-04/assets/8871767/ba9ac29f-f1a2-4ff1-a4bf-d57a08f93a9c)

### 2. Project Pipline
![수정됨_Untitled 3 (2)](https://github.com/boostcampaitech6/level2-movierecommendation-recsys-04/assets/8871767/f1f781dd-a5ed-4466-a9a1-2ea5e8e14bd0)

### 3. Environment
- 서버 정보 : AI Stages GPU V100 서버
- 버전 정보 : Python 3.10.13
- 패키지 정보
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
### 4. Data
#### **train data**
- 주 학습 데이터로 userid*,* itemid, timestamp로 구성되어있으며, 총 **5,154,471**의 행으로 이루어졌다.
- `userid` : 총 **31,360** 명의 유저의 userid가 존재
- `itemid` : 총 **6,807** 건의 영화의 itemid가 존재
- `timestamp` : 유저가 영화를 시청한 시간 이력
#### **item data**
- `director` : 영화별 감독에 대한 자료로, 총 **5,905**개의 행으로 이루어져있다.
- `writer` : 영화별 작가에 대한 자료로, 총 **11,307**개의 행으로 이루어져있다.
- `genre` : 영화의 장르 (한 영화에 여러 장르가 포함될 수 있음)에 대한 자료로, 총 **15,934**개의 행, 총 **18개의 장르**로 이루어져 있다.
- `year` : 영화의 개봉년도에 대한 자료로, 총 **6,799**개의 행으로 이루어져있다.
- `title` : 영화의 제목에 대한 자료로, 총 **6,807**개의 행으로 이루어져 있다.

### 5. Model Performance
| Type | Model | Public Score | Private Score |
| --- | --- | --- | --- |
| General | EASE | 0.1595 | 0.1595 |
|  | ADMMSLIM | 0.1563 | 0.1544 |
|  | CDAE | 0.1318 | 0.1328 |
|  | MultiVAE | 0.1304 | 0.1320 |
|  | MultiDAE | 0.1371 | 0.1390 |
|  | RecVAE | 0.1321 | 0.1335 |
| Context-Aware | DeepFM | 0.0880 | 0.0877 |
| Sequence | SASRec | 0.0949 | 0.0805 |
|  | Bert4Rec | 0.0782 | 0.0738 |
| Graph | LightGCN | 0.1160 | 0.1184 |

### 6. Final submission
Hard Voting 앙상블 : EASE(3), ADMMSLIM(3), CDAE(1), MultiVAE(1), MultiDAE(1), DeepFM(1), SASRec(1), LightGCN(1), Bert4Rec(1)
| Public Score | Private Score |
| --- | --- |
| 0.1632 | 0.1623 |

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

# Laboratory Report
[Movie Recommendation Wrap-Up Report]
