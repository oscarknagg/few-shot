import torch

from few_shot.models import Bottleneck
from few_shot.metrics import NAMED_METRICS


def evaluate(model, dataloader, prepare_batch, metrics, loss_fn=None, prefix='val_', suffix=''):
    logs = {}
    seen = 0
    totals = {m: 0 for m in metrics}
    if loss_fn is not None:
        totals['loss'] = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = prepare_batch(batch)
            y_pred = model(x)

            seen += x.shape[0]

            if loss_fn is not None:
                totals['loss'] += loss_fn(y_pred, y).item() * x.shape[0]

            for m in metrics:
                if isinstance(m, str):
                    v = NAMED_METRICS[m](y, y_pred)
                else:
                    # Assume metric is a callable function
                    v = m(y, y_pred)

                totals[m] += v * x.shape[0]

    for m in ['loss'] + metrics:
        logs[prefix + m + suffix] = totals[m] / seen

    return logs
