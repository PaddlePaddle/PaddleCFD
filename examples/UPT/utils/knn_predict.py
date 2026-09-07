import einops
import paddle
from paddle_utils import *


@paddle.no_grad()
def knn_predict(
    train_x, train_y, test_x, k=10, tau=0.07, batch_normalize=True, eps=1e-06
):
    n_classes = train_y._max().item() + 1
    if n_classes <= 1:
        return None
    class_onehot = paddle.diag(paddle.ones(n_classes, device=train_x.device))
    if batch_normalize:
        mean = train_x.mean(dim=0)
        std = train_x.std(axis=0) + eps
        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std
    train_x = paddle.nn.functional.normalize(train_x, dim=1)
    test_x = paddle.nn.functional.normalize(test_x, dim=1)
    k = min(k, len(train_x))
    similarities = test_x @ train_x.T
    topk_similarities, topk_indices = similarities.topk(k=k, dim=1)
    flat_topk_indices = einops.rearrange(topk_indices, "n_test knn -> (n_test knn)")
    flat_nn_labels = train_y[flat_topk_indices]
    flat_nn_onehot = class_onehot[flat_nn_labels]
    nn_onehot = einops.rearrange(
        flat_nn_onehot, "(n_test k) n_classes -> k n_test n_classes", k=k
    )
    topk_similarities = (topk_similarities / tau).exp_()
    logits = (
        nn_onehot * einops.rearrange(topk_similarities, "n_test knn -> knn n_test 1")
    ).sum(dim=0)
    knn_classes = logits.argmax(dim=1)
    return knn_classes
