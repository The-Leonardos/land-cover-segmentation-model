import torch
import segmentation_models_pytorch as smp
from landcover.utils.util import compute_iou

def train(model_instance, data_loader, opt, loss_fn, device="cpu"):
    model_instance.train()
    total_tp = total_fp = total_fn = total_tn = None
    running_loss = 0

    for images, masks in data_loader:
        images = images.to(device)
        masks = masks.to(device)

        opt.zero_grad()

        outputs = model_instance(images)

        loss = loss_fn(outputs, masks)
        tp, fp, fn, tn = compute_iou(outputs, masks)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 1.0)
        opt.step()

        running_loss += loss.item()

        if total_tp is None:
            total_tp = tp
            total_fp = fp
            total_fn = fn
            total_tn = tn
        else:
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    avg_loss = running_loss / len(data_loader)
    m_iou = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn, reduction="macro")

    return avg_loss, m_iou.item()