import torch
import torch.nn.functional as F


def soft_cross_entropy_sparse(logits, label, num_classes=10, reduce_mean=True):
    label = F.one_hot(label, num_classes).type(torch.float)
    logp = logits - torch.logsumexp(logits, -1, keepdim=True)
    ce = -1 * torch.sum(label * logp, -1)
    if reduce_mean is True:
        return torch.mean(ce)
    else:
        return ce


def soft_cross_entropy(logits, label, reduce_mean=True):
    logp = logits - torch.logsumexp(logits, -1, keepdim=True)
    ce = -1 * torch.sum(label * logp, -1)
    if reduce_mean is True:
        return torch.mean(ce)
    else:
        return ce


def soft_binary_cross_entropy(logits, label, reduce_mean=True):
    sp = F.softplus(logits)
    loss = -1 * (logits * label - sp)
    if reduce_mean is True:
        return torch.mean(loss)
    else:
        return loss


def entropy_logits(logits, reduce_mean=True):
    entropy = soft_cross_entropy(logits, F.softmax(logits, dim=-1),
                                 reduce_mean)
    return entropy


def entropy_batch_uniform(logits):
    batch_p = torch.mean(F.softmax(logits, dim=-1), 0, keepdim=True)
    logit_batch_p = torch.log(batch_p)
    target = F.softmax(torch.zeros_like(logit_batch_p), dim=-1)
    return soft_cross_entropy(logit_batch_p, target)


def entropy_minimization_SSL(logits, min_weight, uni_weight):
    entropy_min = entropy_logits(logits)
    entropy_uni = entropy_batch_uniform(logits)
    return min_weight * entropy_min + uni_weight * entropy_uni
