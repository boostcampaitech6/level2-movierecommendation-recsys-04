hyper_tuning.py

현재 파이썬의 버전과 intergers 명령어가 호환이 되지 않습니다.
따라서 해당하는 부분을 변경하여 run_hyper가 구동되는 것을 확인했습니다.

경로는 opt > conda > lib > python3.8 > site_packaghes > hyperopt > pyll > stochastic.py > randint 입니다.
- 본인의 conda가 설치된 위치에서 hyperopt안의 stochastic.py를 덮어씌우시면 됩니다.
변경된 코드는 대략 127번째 줄입니다.
# 주의
- 혹시 모를 상황에 대비해서 원본파일을 백업해두시기 바랍니다.
