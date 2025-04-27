from pathlib import Path

import hydra
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

from omegaconf import DictConfig
from tqdm import tqdm
import sys
from functools import partial

from models.utils import EarlyStopping, Tee
from dataset.dataset_ESC50 import ESC50, get_global_stats

global_stats = np.array([[-54.369884,  20.843649],
                         [-54.27712 ,  20.85503 ],
                         [-54.188335 , 20.79954 ],
                         [-54.236378 , 20.792656],
                         [-54.193005 , 20.953829]])

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

    # train_loader gibt ein batch an mel spectrogramm und labels zurück
    # 32 samples in einem batch
    # Zufälliges Auswählen von Samples macht der DataLoader
    for _, x, label in tqdm(train_loader, unit='bat', disable=cfg.training.disable_bat_pbar, position=0):
        # Epoche wird per Batchsize in steps unterteilt
        # Das hier ist quasi ein Step
        # Sehr abhängig von num_workers. Prozesse arbeiten parallel für einen Batch

        # move data to device/GPU
        # 30 milionen Parameter
        x = x.float().to(device)
        y_true = label.to(device)

        # the forward pass through the model
        # model/model_classifier.py/forward function
        y_prob = model(x)

        # we could also use 'F.one_hot(y_true)' for 'y_true', but this would be slower
        # Loss ist ein Scalar / Batch
        loss = criterion(y_prob, y_true)
        # reset the gradients to zero - avoids accumulation
        # (sonst könnte man die Gradienten von mehreren Batches addieren)
        optimizer.zero_grad()
        # compute the gradient with backpropagation
        loss.backward()
        # gradient hat die selbe Dimension wie die Parameter
        losses.append(loss.item())
        # minimize the loss via the gradient - adapts the model parameters
        # Hier passiert der gradient descent
        optimizer.step()

        # Ziel parameter anpassen
        # wenn die Batchsize halbiert wird, gibts mehr steps
        # für die parameter bedeutet das, das wir mehr Änderungen haben
        # aber wir erzeugen auch mehr Rauschen über das gesamte Datenset betrachtet

        y_pred = torch.argmax(y_prob, dim=1)
        corrects += (y_pred == y_true).sum().item()
        samples_count += y_true.shape[0]

    acc = corrects / samples_count
    return acc, losses


def fit_classifier():
    num_epochs = cfg.training.epochs

    loss_stopping = EarlyStopping(
        patience=cfg.training.patience,
        delta=cfg.training.delta,
        verbose=True,
        float_fmt=float_fmt,
        checkpoint_file=os.path.join(experiment, 'best_val_loss.pt')
    )

    pbar = tqdm(range(1, 1 + num_epochs), ncols=50, unit='ep', file=sys.stdout, ascii=True)
    for epoch in (range(1, 1 + num_epochs)):
        # iterate once over training data
        # Erzeugen und zerstören Prozesse (num_workers)
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
        # Passt die Lernrate an
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
    data_path = Path(cfg.data.path)
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    device = torch.device(f"cuda:{cfg.training.device_id}" if use_cuda else "mps" if use_mps else "cpu")

    experiment_root = Path(os.getcwd())
    print(f"Output directory: {experiment_root}")

    print(f"Calculating global mean and std for training data of each fold from: {data_path}")
    # global_stats = get_global_stats(cfg=cfg, data_path=data_path)
    print("Calculation finished. Global stats (mean, std) per fold:")
    print(global_stats)
    print("-" * 30)

    # for all folds
    scores = {}
    for test_fold in cfg.data.test_folds:
        experiment = os.path.join(experiment_root, f'{test_fold}')
        os.makedirs(experiment, exist_ok=True)

        # clone stdout to file (does not include stderr). If used may confuse linux 'tee' command.
        with Tee(os.path.join(experiment, 'train.log'), 'w', 1, encoding='utf-8',
                 newline='\n', proc_cr=True):
            # this function assures consistent 'test_folds' setting for train, val, test splits
            # partial ist eine standardfunktion von Python, womit man einige Argumente einer Funktion
            # vorab festlegen kann. Hier wird der ESC50-Datensatz mit den Testfalten 1-5 geladen.

            # Für jeden Testfold laden wir den ESC50-Datensatz mit den entsprechenden Testfalten.
            get_fold_dataset = partial(
                ESC50,
                root=data_path,
                download=True,
                test_folds={test_fold},
                global_mean_std=global_stats[test_fold - 1],
                cfg=cfg
            )

            # liefer ein data von ESC50 wo man __getitem__ und __len__ aufrufen kann
            # liefer mel spectrogramm
            train_set = get_fold_dataset(subset="train")
            print('*****')
            print(f'train folds are {train_set.train_folds} and test fold is {train_set.test_folds}')

            # data loader verwendet config, was angepasst werden kann
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.training.num_workers,
                drop_last=False,
                persistent_workers=cfg.training.persistent_workers,
                pin_memory=True,
            )

            val_loader = torch.utils.data.DataLoader(
                get_fold_dataset(subset="val"),
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                drop_last=False,
                persistent_workers=cfg.training.persistent_workers,
            )

            # instantiate model
            model = hydra.utils.instantiate(cfg.model)
            model = model.to(device)
            print('*****')

            # Define a loss function and optimizer
            criterion = nn.CrossEntropyLoss().to(device)

            if cfg.training.optimizer.name == "adam":
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=cfg.training.optimizer.lr,
                    weight_decay=cfg.training.optimizer.weight_decay
                )
            elif cfg.training.optimizer.name == "sgd":
                # SGD optimizer with momentum and weight decay
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=cfg.training.optimizer.lr,
                    momentum=cfg.training.optimizer.momentum,
                    weight_decay=cfg.training.optimizer.weight_decay
                )

            # nach fixer Anzahl an Epochen wird die Lernrate angepasst
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=cfg.training.step_size,
                gamma=cfg.training.gamma
            )

            # fit the model using only training and validation data, no testing data allowed here
            print()
            fit_classifier()

            # tests
            test_loader = torch.utils.data.DataLoader(
                get_fold_dataset(subset="test"),
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                drop_last=False,
            )

            print(f'\ntest {experiment}')
            test_acc, test_loss, _ = test(
                model,
                test_loader,
                criterion=criterion,
                device=device
            )
            scores[test_fold] = pd.Series(dict(TestAcc=test_acc, TestLoss=np.mean(test_loss)))
            print(scores[test_fold])
            print()

    scores = pd.concat(scores).unstack([-1])
    print(pd.concat((scores, scores.agg(['mean', 'std']))))


if __name__ == "__main__":
    # digits for logging
    float_fmt = ".3f"
    pd.options.display.float_format = ('{:,' + float_fmt + '}').format
    main()