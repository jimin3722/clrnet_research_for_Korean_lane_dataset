import os
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import random
from clrnet.utils.config import Config
from clrnet.engine.runner import Runner
from clrnet.datasets import build_dataloader


def main():
    args = parse_args()

    # 환경 변수 "CUDA_VISIBLE_DEVICES"를 설정하여 사용 가능한 GPU를 지정합니다. 특히, args.gpus는 GPU ID 목록을 포함하며, 코드는 이 목록을 문자열로 변환하여 쉼표로 연결합니다.
    # 예를 들어, args.gpus가 [0, 2]라면, 이 코드는 "CUDA_VISIBLE_DEVICES"를 "0,2"로 설정하여 첫 번째와 세 번째 GPU만 사용하도록 만듭니다.
    # 이 설정은 딥 러닝 라이브러리 (예: TensorFlow, PyTorch)가 사용하는 GPU를 제한하는 데 사용되며, 특정 GPU만 사용하도록 제어할 수 있게 해줍니다.
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(gpu) for gpu in args.gpus)

    cfg = Config.fromfile(args.config)
    cfg.gpus = len(args.gpus)
    cfg.load_from = args.load_from
    cfg.resume_from = args.resume_from
    cfg.finetune_from = args.finetune_from
    cfg.view = args.view
    cfg.seed = args.seed

    cfg.work_dirs = args.work_dirs if args.work_dirs else cfg.work_dirs
    cudnn.benchmark = True

    runner = Runner(cfg)

    if args.validate:
        runner.validate()
    elif args.test:
        runner.test()
    else:
        runner.train()


def parse_args():

    # 'config': 훈련 설정 파일의 경로입니다.
    # '--work_dirs': 작업 디렉터리의 경로입니다.
    # '--load_from': 로드할 체크포인트 파일의 경로입니다.
    # '--resume_from': 이어서 학습을 시작할 체크포인트 파일의 경로입니다.
    # '--finetune_from': 미세 조정을 시작할 체크포인트 파일의 경로입니다.
    # '--view': 이 인수가 제공되면 뷰 옵션이 활성화됩니다.
    # '--validate': 이 인수가 제공되면 훈련 중 체크포인트를 평가합니다.
    # '--test': 이 인수가 제공되면 테스트 세트에서 체크포인트를 테스트합니다.
    # '--gpus': 사용할 GPU의 ID 목록입니다.
    # '--seed': 랜덤 시드 값입니다. 같은 시드 값을 사용하면 실험의 재현성을 보장할 수 있습니다.
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dirs',
                        type=str,
                        default=None,
                        help='work dirs')
    parser.add_argument('--load_from',
                        default=None,
                        help='the checkpoint file to load from')
    parser.add_argument('--resume_from',
            default=None,
            help='the checkpoint file to resume from')
    parser.add_argument('--finetune_from',
            default=None,
            help='the checkpoint file to resume from')
    parser.add_argument('--view', action='store_true', help='whether to view')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test',
        action='store_true',
        help='whether to test the checkpoint on testing set')
    parser.add_argument('--gpus', nargs='+', type=int, default='0')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
