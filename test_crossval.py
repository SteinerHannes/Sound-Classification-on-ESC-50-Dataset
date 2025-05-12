import argparse
import os

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from dataset.dataset_ESC50 import ESC50, download_extract_zip, get_global_stats
from train_crossval import test, use_backup_config, use_hydra


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    reproducible = cfg.test.reproducible
    data_path = cfg.data.path
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    device = torch.device(f"cuda:{cfg.training.device_id}" if use_cuda else "mps" if use_mps else "cpu")

    if reproducible:
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)

    experiment_root = cfg.test.experiment_path
    if not os.path.isdir(experiment_root):
        print('Downloading model parameters...')
        download_extract_zip(
            url='https://cloud.technikum-wien.at/s/PiHsFtnB69cqxPE/download/sample-run.zip',
            file_path=experiment_root + '.zip',
        )

    print(f"Calculating global mean and std for training data of each fold from: {data_path}")
    global_stats = get_global_stats(cfg=cfg, data_path=data_path)
    print("Calculation finished. Global stats (mean, std) per fold:")
    print(global_stats)
    print("-" * 30)

    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    scores = {}
    probs = {model_file_name: {} for model_file_name in cfg.test.checkpoints}
    for test_fold in cfg.data.test_folds:
        experiment = os.path.join(experiment_root, f'{test_fold}')

        test_loader = torch.utils.data.DataLoader(
            ESC50(
                cfg=cfg,
                subset="test",
                test_folds={test_fold},
                global_mean_std=global_stats[test_fold - 1],
                root=data_path,
                download=True
            ),
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            drop_last=False,
        )

        scores[test_fold] = {}
        for model_file_name in cfg.test.checkpoints:
            model_file = os.path.join(experiment, model_file_name)
            sd = torch.load(model_file, map_location=device)
            model.load_state_dict(sd)
            print('Testing', model_file)
            test_acc, test_loss, p = test(cfg, model, test_loader, criterion=criterion, device=device)
            probs[model_file_name].update(p)
            scores[test_fold][model_file_name] = pd.Series(dict(TestAcc=test_acc, TestLoss=np.mean(test_loss)))
            print(scores[test_fold][model_file_name])
        scores[test_fold] = pd.concat(scores[test_fold])
        scores[test_fold].to_csv(os.path.join(experiment, 'test_scores.csv'),
                                 index_label=['checkpoint', 'metric'], header=['value'])

    scores = pd.concat(scores).unstack([-2, -1])
    scores = pd.concat((scores, scores.agg(['mean', 'std'])))
    for model_file_name in cfg.test.checkpoints:
        file_name = os.path.splitext(model_file_name)[0]
        file_path = os.path.join(experiment_root, f'test_probs_{file_name}.csv')
        probs[model_file_name] = pd.DataFrame(probs[model_file_name]).T
        probs[model_file_name].to_csv(file_path)
        file_path = os.path.join(experiment_root, f'test_scores_{file_name}.csv')
        scores[model_file_name].to_csv(file_path)
    print(scores)
    print()


if __name__ == "__main__":
    # digits for logging
    float_fmt = ".3f"
    pd.options.display.float_format = ('{:,' + float_fmt + '}').format
    if use_hydra:
        main()
    else:
        cfg = use_backup_config()
        parser = argparse.ArgumentParser()
        parser.add_argument('cvpath', nargs='?', default=cfg.test.experiment_path)
        args = parser.parse_args()
        cfg.test.experiment_path = args.cvpath
        main(cfg)