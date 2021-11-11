import torch
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_curve, auc

from src.utils import torch_onehot
from src.evaluation.utils import cat_callibration, expected_callibration_error


def class_brier(y, log_probs=None, probs=None):
    assert log_probs is None or probs is None
    if log_probs is not None:
        probs = log_probs.exp()
    elif probs is not None:
        pass
    else:
        raise Exception('either log_probs or probs must not be None')
    if not (probs.max().item() <= (1. + 1e-4) and probs.min().item() >= 0.):
        print(f"probs.max(): {probs.max().item()}, probs.min(): {probs.min().item()}")
    probs = torch.clamp(probs, min=0., max=1.)
    assert probs.max().item() <= (1. + 1e-4) and probs.min().item() >= 0.
    if len(y.shape) > 1:
        y = y.squeeze(1)
    y_oh = torch_onehot(y, probs.shape[1])
    brier = (probs - y_oh).pow(2).sum(dim=1).mean(dim=0)
    return brier.item()


def class_err(y, model_out):
    pred = model_out.max(dim=1, keepdim=False)[1]  # get the index of the max probability
    err = pred.ne(y.data).sum().item() / y.shape[0]
    return err


def class_ll(y, log_probs=None, probs=None, eps=1e-40):
    assert log_probs is None or probs is None
    if log_probs is not None:
        pass
    elif probs is not None:
        log_probs = torch.log(probs.clamp(min=eps))
    else:
        raise Exception('either log_probs or probs must not be None')
    nll = F.nll_loss(log_probs, y, reduction='mean')
    return -nll.item()


def class_ECE(y, log_probs=None, probs=None, nbins=10, top_k=1):
    assert log_probs is None or probs is None
    if log_probs is not None:
        probs = log_probs.exp()
    elif probs is not None:
        pass
    else:
        raise Exception('either log_probs or probs must not be None')
    if not (probs.max().item() <= (1. + 1e-4) and probs.min().item() >= 0.):
        print(f"probs.max(): {probs.max().item()}, probs.min(): {probs.min().item()}")
    probs = torch.clamp(probs, min=0., max=1.)
    assert probs.max().item() <= (1. + 1e-4) and probs.min().item() >= 0.
    probs = probs.clamp(max=(1-1e-8))
    bin_probs, _, _, bin_counts, reference = cat_callibration(probs.cpu().numpy(), y.cpu().numpy(),
                                                              nbins, top_k=top_k)
    ECE = expected_callibration_error(bin_probs, reference, bin_counts)
    return ECE


def entropy_from_logprobs(log_probs):
    return - (log_probs.exp() * log_probs).sum(dim=1)


def get_roc_params(ID_entropy, OOD_entropy):
    if torch.is_tensor(ID_entropy):
        ID_entropy = ID_entropy.data.cpu().numpy()
    if torch.is_tensor(OOD_entropy):
        OOD_entropy = OOD_entropy.data.cpu().numpy()

    targets = np.concatenate([np.ones(OOD_entropy.shape[0]), np.zeros(ID_entropy.shape[0])], axis=0)
    entropy_vals = np.concatenate([OOD_entropy, ID_entropy], axis=0)
    fpr, tpr, _ = roc_curve(targets, entropy_vals)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def classification_rejection(logprob_vec, source_labels_vec, source_entropy, target_entropy, rejection_step=0.005):

    pred = logprob_vec.max(dim=1, keepdim=False)[1]  # get the index of the max probability
    err_vec_in = pred.ne(source_labels_vec.data).cpu().numpy()
    err_vec_out = np.ones(target_entropy.shape[0])

    full_err_vec = np.concatenate([err_vec_in, err_vec_out], axis=0)
    full_entropy_vec = np.concatenate([source_entropy, target_entropy], axis=0)
    sort_entropy_idxs = np.argsort(full_entropy_vec, axis=0)
    Npoints = sort_entropy_idxs.shape[0]

    err_props = []

    for rej_prop in np.arange(0, 1, rejection_step):
        N_reject = np.round(Npoints * rej_prop).astype(int)
        if N_reject > 0:
            accepted_idx = sort_entropy_idxs[:-N_reject]
        else:
            accepted_idx = sort_entropy_idxs

        err_props.append(full_err_vec[accepted_idx].sum() / (accepted_idx.shape[0] + 1e-40))

        if not (err_props[-1].max() <= 1 and err_props[-1].min() >= 0):
            print(f"err_props[-1].max(): {err_props[-1].max()}, err_props[-1].min(): {err_props[-1].min()}")
        err_props[-1] = np.clip(err_props[-1], a_min=0, a_max=1)
        assert err_props[-1].max() <= 1 and err_props[-1].min() >= 0

    return np.array(err_props)


def get_all_classification_stats(logprob_vec, target_vec, target_logrobs):

    brier = class_brier(y=target_vec, log_probs=logprob_vec, probs=None)
    err = class_err(y=target_vec, model_out=logprob_vec)
    ll = class_ll(y=target_vec, log_probs=logprob_vec, probs=None, eps=1e-40)
    ece = class_ECE(y=target_vec, log_probs=logprob_vec, probs=None, nbins=10)
    return_dict = {"err": err, "ll": ll, "brier": brier, "ece": ece}

    if target_logrobs is not None:
        source_entropy = entropy_from_logprobs(logprob_vec).cpu().numpy()
        target_entropy = entropy_from_logprobs(target_logrobs).cpu().numpy()

        fpr, tpr, roc_auc = get_roc_params(source_entropy, target_entropy)

        classification_rejection_err = classification_rejection(logprob_vec, target_vec, source_entropy, target_entropy, rejection_step=0.005)

        return_dict.update({"roc_auc": roc_auc, "cr_err": classification_rejection_err, "fpr": fpr, "tpr": tpr})

    return return_dict
