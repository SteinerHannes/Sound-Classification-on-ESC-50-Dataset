import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader
import pandas as pd
import numpy as np
import os
import datetime
from pathlib import Path
from tqdm import tqdm
import sys
from functools import partial

from models.model_classifier import AudioMLP
from models.utils import EarlyStopping, Tee
from dataset.dataset_ESC50 import ESC50
# No longer allowed: import config


# evaluate model on different testing data 'dataloader'
def test(model, dataloader, criterion, device):
    model.eval()

    losses = []
    corrects = 0
    samples_count = 0
    probs = {}
    with torch.no_grad():
        # no gradient computation needed
        for k, x, label in tqdm(dataloader, unit='bat', disable=cfg.training.disable_bat_pbar, position=0):
            x = x.float().to(device)
            y_true = label.to(device)

            # the forward pass through the model
            y_prob = model(x)

            loss = criterion(y_prob, y_true)
            losses.append(loss.item())

            y_pred = torch.argmax(y_prob, dim=1)
            corrects += (y_pred == y_true).sum().item()
            samples_count += y_true.shape[0]
            for w, p in zip(k, y_prob):
                probs[w] = [float(v) for v in p]

    acc = corrects / samples_count
    return acc, losses, probs


def train_epoch():
    # switch to training
    model.train()

    losses = []
    corrects = 0
    samples_count = 0
    for _, x, label in tqdm(train_loader, unit='bat', disable=cfg.training.disable_bat_pbar, position=0):
        x = x.float().to(device)
        y_true = label.to(device)

        # the forward pass through the model
        y_prob = model(x)

        # we could also use 'F.one_hot(y_true)' for 'y_true', but this would be slower
        loss = criterion(y_prob, y_true)
        # reset the gradients to zero - avoids accumulation
        optimizer.zero_grad()
        # compute the gradient with backpropagation
        loss.backward()
        losses.append(loss.item())
        # minimize the loss via the gradient - adapts the model parameters
        optimizer.step()

        y_pred = torch.argmax(y_prob, dim=1)
        corrects += (y_pred == y_true).sum().item()
        samples_count += y_true.shape[0]

    acc = corrects / samples_count
    return acc, losses


def fit_classifier():
    num_epochs = cfg.training.epochs

    loss_stopping = EarlyStopping(patience=cfg.training.patience, delta=0.002, verbose=True, float_fmt=float_fmt,
                                  checkpoint_file=os.path.join(experiment, 'best_val_loss.pt'))

    pbar = tqdm(range(1, 1 + num_epochs), ncols=50, unit='ep', file=sys.stdout, ascii=True)
    for epoch in (range(1, 1 + num_epochs)):
        # iterate once over training data
        train_acc, train_loss = train_epoch()

        # validate model
        val_acc, val_loss, _ = test(model, val_loader, criterion=criterion, device=device)
        val_loss_avg = np.mean(val_loss)

        # print('\n')
        pbar.update()
        # pbar.refresh() syncs output when pbar on stderr
        # pbar.refresh()
        print(end=' ')
        print(  # f" Epoch: {epoch}/{num_epochs}",
            f"TrnAcc={train_acc:{float_fmt}}",
            f"ValAcc={val_acc:{float_fmt}}",
            f"TrnLoss={np.mean(train_loss):{float_fmt}}",
            f"ValLoss={val_loss_avg:{float_fmt}}",
            end=' ')

        early_stop, improved = loss_stopping(val_loss_avg, model, epoch)
        if not improved:
            print()
        if early_stop:
            print("Early stopping")
            break

        # advance the optimization scheduler
        scheduler.step()
    # save full model
    torch.save(model.state_dict(), os.path.join(experiment, 'terminal.pt'))


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    # make config global for this file only, as quick temporary solution.
    # You should better pass config as function arguments.
    global cfg
    cfg = config

    global model, train_loader, val_loader, criterion, optimizer, scheduler, device, experiment
    data_path = Path(config.data.path)
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{config.training.device_id}" if use_cuda else "cpu")

    #experiment_root = os.path.join(runs_path, str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')))
    experiment_root = Path(os.getcwd())
    print(f"Output directory: {experiment_root}")
    #if not os.path.exists(experiment_root):
    #    os.mkdir(experiment_root)

    # for all folds
    scores = {}
    for test_fold in config.data.test_folds:
        experiment = os.path.join(experiment_root, f'{test_fold}')
        os.makedirs(experiment, exist_ok=True)

        # clone stdout to file (does not include stderr). If used may confuse linux 'tee' command.
        with Tee(os.path.join(experiment, 'train.log'), 'w', 1, encoding='utf-8',
                 newline='\n', proc_cr=True):
            # this function assures consistent 'test_folds' setting for train, val, test splits
            get_fold_dataset = partial(ESC50, test_folds={test_fold}, root=data_path, download=True,
                                       # you have to add this argument to ESC50
                                       config=config.data)

            train_set = get_fold_dataset(subset="train")
            print('*****')
            print(f'train folds are {train_set.train_folds} and test fold is {train_set.test_folds}')
            print('random wave cropping')

            train_loader = torch.utils.data.DataLoader(train_set,
                                                       batch_size=config.training.batch_size,
                                                       shuffle=True,
                                                       num_workers=config.training.num_workers,
                                                       drop_last=False,
                                                       persistent_workers=config.training.persistent_workers,
                                                       pin_memory=True,
                                                       )

            val_loader = torch.utils.data.DataLoader(get_fold_dataset(subset="val"),
                                                     batch_size=config.training.batch_size,
                                                     shuffle=False,
                                                     num_workers=config.training.num_workers,
                                                     drop_last=False,
                                                     persistent_workers=config.training.persistent_workers,
                                                     )

            print()
            # instantiate model
            model = hydra.utils.instantiate(config.model)
            # model = nn.DataParallel(model, device_ids=config.device_ids)
            model = model.to(device)
            print('*****')

            # Define a loss function and optimizer
            criterion = nn.CrossEntropyLoss().to(device)

            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=config.training.lr,
                                        momentum=0.9,
                                        weight_decay=config.training.weight_decay)

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=config.training.step_size,
                                                        gamma=config.training.gamma)

            # fit the model using only training and validation data, no testing data allowed here
            print()
            fit_classifier()

            # tests
            test_loader = torch.utils.data.DataLoader(get_fold_dataset(subset="test"),
                                                      batch_size=config.training.batch_size,
                                                      shuffle=False,
                                                      num_workers=0,  # config.num_workers,
                                                      drop_last=False,
                                                      )

            print(f'\ntest {experiment}')
            test_acc, test_loss, _ = test(model, test_loader, criterion=criterion, device=device)
            scores[test_fold] = pd.Series(dict(TestAcc=test_acc, TestLoss=np.mean(test_loss)))
            print(scores[test_fold])
            # print(scores[test_fold].unstack())
            print()
    scores = pd.concat(scores).unstack([-1])
    print(pd.concat((scores, scores.agg(['mean', 'std']))))


if __name__ == "__main__":
    # digits for logging
    float_fmt = ".3f"
    #experiment = None
    main()
