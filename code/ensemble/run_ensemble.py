import os
import numpy as np
import argparse

from voting import Ensemble


def get_file_list(directory):
    file_list = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_list.append(filename[:-4])
    return file_list


def main(args):
    if args.file_path:
        files = get_file_list(args.file_path)
    else:
        files = args.files[0]
    if args.weight is None:  # 가중치를 주지 않았을 경우
        w_list = [1 / len(files)] * len(files)  # 동일한 가중치 부여
    else:
        w_list = args.weight[0]

    en = Ensemble(files, args.file_path)

    os.makedirs(args.file_path, exist_ok=True)  # 읽어들일 파일 경로
    os.makedirs(args.result_path, exist_ok=True)  # 결과 파일 저장 경로

    if os.listdir(args.file_path) == []:
        raise ValueError(f"앙상블 할 파일을 {args.file_path}에 넣어주세요.")
    if len(files) < 2:
        raise ValueError("2개 이상의 모델이 필요합니다.")
    if not len(files) == len(w_list):
        raise ValueError("model과 weight의 길이가 일치하지 않습니다.")

    # 앙상블 전략에 따라 수행
    if args.strategy == "hard":
        strategy_title = "hard"
        result = en.hard_voting()
    elif args.strategy == "soft":
        strategy_title = "soft-" + "-".join(
            map(str, ["{:.2f}".format(w) for w in w_list])
        )
        result = en.soft_voting(w_list)
    else:
        raise ValueError(
            '[hard, soft] 중 앙상블 전략을 선택해 주세요. (default="hard")'
        )

    # 결과 저장
    if args.result_fname is None:
        files_title = "+".join(files)
    else:
        files_title = args.result_fname

    save_file_path = f"{args.result_path}{strategy_title}({files_title}).csv"
    result.to_csv(save_file_path, index=False)
    print(f"The result is in {save_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser")
    arg = parser.add_argument

    arg(
        "--file_path",
        "-fp",
        type=str,
        default="./ensembles/",
        help="앙상블 파일 열기 경로",
    )
    arg(
        "--result_path",
        "-rp",
        type=str,
        default="./submit/",
        help="앙상블 결과 저장 경로",
    )
    arg(
        "--files",
        "-f",
        nargs="+",
        default=None,
        type=lambda s: [item for item in s.split(",")],
        help="앙상블 파일명(쉼표 구분)",
    )
    arg(
        "--weight",
        "-w",
        nargs="+",
        default=None,
        type=lambda s: [float(item) for item in s.split(",")],
        help="앙상블 모델 가중치 설정",
    )
    arg(
        "--strategy",
        "-s",
        type=str,
        default="soft",
        choices=["hard", "soft"],
        help="앙상블 전략 선택",
    )

    args = parser.parse_args()
    main(args)
