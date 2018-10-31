import torch


def categorical_accuracy(y, y_pred):
    """Calculates categorical accuracy.

    `y_pred` have shape (batch_size, num_categories,)
    `y` must have shape (batch_size,)
    """
    return torch.eq(y_pred.argmax(dim=-1), y).sum().item() / y_pred.shape[0]


NAMED_METRICS = {
    'categorical_accuracy': categorical_accuracy
}
