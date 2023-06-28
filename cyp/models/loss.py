import torch.nn.functional as F
import torch
import tllib


def flatten_outputs(fea):
    return torch.reshape(fea, (fea.shape[0], fea.shape[1], fea.shape[2] * fea.shape[3]))


def l1_l2_loss(pred, true, l1_weight, scores_dict, l2_sp_weight=0, l2_sp_beta=0, bss=0, model=None, sp_model=None,
               x=None, k=0, delta=0, layer_outputs_target=None, input=None, delta_w_att=0, channel_weights=None):
    """
    Regularized MSE loss; l2 loss with l1 loss too.

    Parameters
    ----------
    pred: torch.floatTensor
        The model predictions
    true: torch.floatTensor
        The true values
    l1_weight: int
        The value by which to weight the l1 loss
    scores_dict: defaultdict(list)
        A dict to which scores can be appended.
    l2_sp_weight: int
        L2-SP Regularization
    bss: int
        If > 0 Batch Spectral Shrinkage will be used (see: https://github.com/thuml/Batch-Spectral-Shrinkage).
        The parameter bss is the parameter for penalty.
    model: Path
        Current model (self.model in base), necessary for l2_sp and bss.
    sp_model: Path
        L2-SP Model as Starting Point

    Returns
    ----------
    loss: the regularized mse loss
    """
    loss = F.mse_loss(pred, true)

    scores_dict["l2"].append(loss.item())

    if l1_weight > 0:
        l1 = F.l1_loss(pred, true)
        loss += l1
        scores_dict["l1"].append(l1.item())
    if l2_sp_weight > 0:
        sp = tllib.regularization.SPRegularization(sp_model.convblocks, model.convblocks)
        loss += l2_sp_weight * sp()
        # l2_norm = sum(p.pow(2.0).sum() for p in model.convblocks[4].parameters())
        # l2_norm += sum(p.pow(2.0).sum() for p in model.convblocks[5].parameters())
        l2_norm = sum(p.pow(2.0).sum() for p in model.dense_layers.parameters())
        loss += (l2_norm/2) * l2_sp_beta
    if bss > 0:
        res = 0
        # for block in model.convblocks:
        #     x = block(x)
        # x = x.view(x.shape[0], -1)
        u, s, v = torch.svd(x.t())
        num = s.size(0)
        if (num - k) < 1:
            k = 0
        for i in range(k):
            res += torch.pow(s[num-1-i], 2)
        loss += bss * res
    if delta > 0:
        layer_outputs_source = {}
        for block in sp_model.convblocks:
            input = block(input)
            layer_outputs_source[block] = input

        delta_out = 0.0
        for fm_src, fm_tgt in zip(layer_outputs_source.values(), layer_outputs_target.values()):
            delta_out += 0.5 * (torch.norm(fm_tgt - fm_src.detach()) ** 2)
        loss += delta_out

    if delta_w_att > 0:
        layer_outputs_source = {}
        for block in sp_model.convblocks:
            input = block(input)
            layer_outputs_source[block] = input

        delta_att = 0.0
        for i, (fm_src, fm_tgt) in enumerate(zip(layer_outputs_source.values(), layer_outputs_target.values())):
            b, c, h, w = fm_src.shape
            fm_src = fm_src.reshape(b, c, h * w)
            fm_tgt = fm_tgt.reshape(b, c, h * w)

            distance = torch.norm(fm_tgt - fm_src.detach(), 2, 2)
            distance = c * torch.mul(channel_weights[i], distance ** 2) / (h * w)
            delta_att += delta_w_att * 0.5 * torch.sum(distance)
        loss += delta_att
        # l2_norm = sum(p.pow(2.0).sum() for p in model.dense_layers.parameters())
        # loss += (l2_norm / 2) * l2_sp_beta

    scores_dict["loss"].append(loss.item())

    return loss, scores_dict


def l2_sp_loss(pred, true, model, sp_model, scores_dict):
    """
    Regularized MSE loss; l2 loss with l1 loss too.

    Parameters
    ----------
    pred: torch.floatTensor
        The model predictions
    true: torch.floatTensor
        The true values
    model: torch.nn.Module
        TCurrent Model.
    sp_model: torch.nn.Module
        The source (starting point) model for L2-SP regularization
    scores_dict: defaultdict(list)
        A dict to which scores can be appended.

    Returns
    ----------
    loss: the regularized mse loss
    """
    loss = F.mse_loss(pred, true)

    # sp = tllib.regularization.SPRegularization(sp_model.convblocks, model.convblocks)
    sp = tllib.regularization.SPRegularization(sp_model.dense_layers, model.dense_layers)
    loss += sp()

    scores_dict["l2"].append(loss.item())
    scores_dict["loss"].append(loss.item())

    return loss, scores_dict
