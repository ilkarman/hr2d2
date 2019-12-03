import torch
from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
import torch.nn.functional as F
from toolz import curry
from torch.nn import functional as F
import numpy as np


def create_supervised_trainer(
    model,
    optimizer,
    loss_fn,
    prepare_batch,
    device=None,
    non_blocking=False,
    output_transform=lambda x, y, preds, loss: {"loss": loss.item()},
):
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        # we're interested in the indices on the max values, not the values themselves

        # _, preds = torch.max(y_pred, 1)  
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return Engine(_update)


@curry
def val_transform(x, y, preds):
    return {"batch": x, "preds": preds.detach(), "labels": y.detach()}


def create_supervised_evaluator(
    model, prepare_batch, metrics=None, device=None, non_blocking=False, output_transform=val_transform,
):
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            # we're interested in the indices on the max values, not the values themselves
            # _, preds = torch.max(y_pred, 1)
            return output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

