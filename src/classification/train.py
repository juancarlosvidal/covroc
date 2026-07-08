"""Generic cross-validated MLP regression trainer (RegressionModel + weighted MSE).

cv_loop() is the shared training entrypoint reused by croc_nn_baseline.py to fit the
mean and residual-variance networks of the two-stage location-scale aROC estimator,
and, run standalone, doubles as the classification-with-temperature-scaling pipeline
(calibration of the FNN's predictive outputs via ModelWithTemperature).
"""
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from data import CustomDataset
from metrics import compute_regression_metrics, dict_mean
from models import RegressionModel
from losses import CustomMSELoss
from temperature_scaling import ModelWithTemperature

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Meter:
    """
    A little helper class which keeps track of statistics during an epoch.
    """

    def __init__(self, name, cum=False):
        """
        name (str or iterable): name of values for the meter
            If an iterable of size n, updates require a n-Tensor
        cum (bool): is this meter for a cumulative value (e.g. time)
            or for an averaged value (e.g. loss)? - default False
        """
        self.cum = cum
        if type(name) == str:
            name = (name,)
        self.name = name

        self._total = torch.zeros(len(self.name))
        self._last_value = torch.zeros(len(self.name))
        self._count = 0.0

    def update(self, data, n=1):
        """
        Update the meter
        data (Tensor, or float): update value for the meter
            Size of data should match size of ``name'' in the initialized args
        """
        self._count = self._count + n
        if torch.is_tensor(data):
            self._last_value.copy_(data)
        else:
            self._last_value.fill_(data)
        self._total.add_(self._last_value)

    def value(self):
        """
        Returns the value of the meter
        """
        if self.cum:
            return self._total
        else:
            return self._total / self._count

    def __repr__(self):
        return '\t'.join(['%s: %.5f (%.3f)' % (n, lv, v)
                          for n, lv, v in zip(self.name, self._last_value, self.value())])


def cv_loop(data, n_folds, n_epochs, batch_size, learning_rate, weight_decay, verbose=False):
    k_fold = KFold(n_splits=n_folds, random_state=42, shuffle=True)
    indexes = sorted(range(len(data['w'])))
    splits = k_fold.split(indexes)

    min_loss = np.inf
    metrics_list = []
    for train_split, test_split in splits:
        # Select the 20% of the train size as the validation set
        train_size = round(len(train_split) * 0.8)
        train_index, val_index, test_index = train_split[:train_size], train_split[train_size:], test_split

        fold_model, fold_min_loss, fold_test_metrics = run_fold(
            data=data,
            train_split=train_index,
            val_split=val_index,
            test_split=test_index,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            verbose=verbose)

        metrics_list.append(fold_test_metrics)
        if fold_min_loss < min_loss:
            model = fold_model

    return model, dict_mean(metrics_list)


def run_fold(data, train_split, val_split, test_split, n_epochs, batch_size, learning_rate, weight_decay, verbose=False):
    min_loss = np.inf

    input, target, weight = data['x'], data['y'], data['w']

    train_input, train_target, train_weight = input[train_split], target[train_split], weight[train_split]
    val_input, val_target, val_weight = input[val_split], target[val_split], weight[val_split]
    test_input, test_target, test_weight = input[test_split], target[test_split], weight[test_split]

    train_loader, val_loader, test_loader = \
        DataLoader(CustomDataset(train_input, train_target, train_weight), batch_size=batch_size), \
        DataLoader(CustomDataset(val_input, val_target, val_weight), batch_size=batch_size), \
        DataLoader(CustomDataset(test_input, test_target, test_weight), batch_size=batch_size)

    model = RegressionModel(input.shape[1])
    model.to(device)

    # Declaring Criterion
    criterion = CustomMSELoss()

    # Declaring Optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

    for epoch in range(1, n_epochs + 1):
        # ####  TRAIN LOOP  #### #
        train_epoch = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            train=True,
            verbose=verbose
        )

        scheduler.step()

        # ####  VALIDATION LOOP  #### #
        val_epoch = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            train=False,
            verbose=False
        )

        # Determine if model is the best
        _, test_loss, _ = val_epoch
        if test_loss[0] < min_loss:
            min_loss = test_loss[0]

    # ####  TEST LOOP  #### #
    _, _, test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epoch=epoch,
        n_epochs=n_epochs,
        train=False,
        verbose=False
    )

    if verbose:
        print('Fold Test MAE: {:.3f} MSE: {:.3f} R2: {:.3f}'.format(
            test_metrics['mae'], test_metrics['mse'], test_metrics['r2']
        ))

    return model, min_loss, test_metrics


def run_epoch(model, loader, criterion, optimizer, epoch=0, n_epochs=0, train=True, verbose=False):
    time_meter = Meter(name='Time', cum=True)
    loss_meter = Meter(name='Loss', cum=False)

    if train:
        model.train()
    else:
        model.eval()

    end = time.time()
    for batch, (input, target, weight) in enumerate(loader):
        # Transfer Data to GPU if available
        input, target, weight = input.to(device), target.to(device), weight.to(device)

        if train:
            # Forward Pass
            output = model(input)
            # Find the Loss
            # loss = criterion(output, target)  # nn.BCELoss()
            loss = criterion(output, target, weight)  # CustomLoss()
            # loss = criterion(pred, torch.squeeze(y))  # nn.CrossEntropyLoss()

            # Backpropagation
            model.zero_grad()
            optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Update weight
            optimizer.step()

        else:
            with torch.no_grad():
                output = model(input)
                # loss = criterion(output, target)  # nn.BCELoss()
                loss = criterion(output, target, weight)  # CustomLoss()

        # Accounting
        # _, predictions = torch.topk(output, 1)
        # error = 1 - torch.eq(predictions, target).float().mean()
        # accuracy = 1 - weighted_binary_accuracy(output, target, weight).item()
        # accuracy = 1 - binary_accuracy(output, target).item()
        batch_time = time.time() - end
        end = time.time()

        # Log errors
        time_meter.update(batch_time)
        loss_meter.update(loss)

        metrics = compute_regression_metrics(weight.detach().cpu().numpy(),
                                             output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())

        if verbose:
            print('{:s}: (Epoch {:d} of {:d}) [{:04d}/{:04d}] {:s} {:s} MAE: {:.3f} MSE: {:.3f} R2: {}'.format(
                'Train' if train else 'Eval', epoch, n_epochs, batch + 1, len(loader), str(time_meter), str(loss_meter),
                metrics['mae'], metrics['mse'], metrics['r2']
            ))

    return time_meter.value(), loss_meter.value(), metrics
