import argparse
import os
import time
from datetime import datetime

import mlconfig
import numpy as np
import torch

import src.criterions   # pylint: disable=unused-import, wrong-import-position
import src.datasets     # pylint: disable=unused-import, wrong-import-position
import src.models       # pylint: disable=unused-import, wrong-import-position
import src.optimizers   # pylint: disable=unused-import, wrong-import-position
import src.schedulers   # pylint: disable=unused-import, wrong-import-position
from src.trainers import Trainer
from src.utils.mlflow_logger import MLFlowLogger
from src.utils.repo import get_repo_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="configure file for training")
    parser.add_argument("-r", "--resume", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Run to debug, i.e. remove the randomness")
    parser.add_argument("--same_run", action="store_true", help="Continue the previous training and state, record in the same run")
    parser.add_argument('--resume_state', action="store_true", help="Load model with the training state")
    parser.add_argument("--local_rank", type=int, default=0, help="GPU id, which is automatically set while distributed data parallel(DDP) training")
    return parser.parse_args()


def manual_seed(seed=0):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prase_id_from_path(path):
    if path is None:
        return None

    # get run_id in model meta
    meta = torch.load(path)["meta"]
    run_id = meta["exp_id"] if "exp_id" in meta else None
    return run_id


def main():
    args = parse_args()
    config = mlconfig.load(args.config)

    if "WORLD_SIZE" in os.environ:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        record_data = (torch.distributed.get_rank() == 0)
    else:
        record_data = True

    if record_data:
        run_id = prase_id_from_path(args.resume) if args.same_run else None
        exp_logger = MLFlowLogger(
            exp_name=config.exp_name,
            tracking_uri=config.record_folder if "record_folder" in config else None,
            run_name=config.run_name,
            run_id=run_id,
        )

        # save the training configuration and parameters
        exp_logger.log_artifact(args.config)
        if "log_params" in config:
            exp_logger.log_params(config.log_params.flat())
        else:
            exp_logger.log_params(config.flat())
    else:
        exp_logger = None

    if args.debug:
        manual_seed()
    else:
        torch.backends.cudnn.benchmark = True

    print(f'cuda:{args.local_rank}')
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

    git_commit, git_tag = get_repo_info()
    meta = {
        "tag": git_tag,
        "version": git_commit,
        "timestamp": datetime.now().strftime(f"UTC{time.timezone / -(60*60):+} %Y-%m-%d, %H:%M:%S"),
        "utctime": datetime.utcnow().strftime("%Y-%m-%d, %H:%M:%S"),
        "exp_id": exp_logger.run_id if exp_logger is not None else "",
    }

    trainer: Trainer = config.trainer(
        meta=meta,
        device=device,
        exp_logger=exp_logger,
        config=config,
    )

    resume_state = args.same_run or args.resume_state
    if args.resume is not None:
        trainer.resume(args.resume, training_state=resume_state)

    trainer.fit()

    if exp_logger is not None:
        exp_logger.close()


if __name__ == '__main__':
    main()
