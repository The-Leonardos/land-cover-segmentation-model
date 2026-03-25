import torch
import segmentation_models_pytorch as smp
from landcover.utils.util import compute_iou, pad_to_multiple, unpad

def test(model_instance, data_loader, loss_fn, patch_size, device="cpu"):
    model_instance.eval()
    total_tp = total_fp = total_fn = total_tn = None
    running_loss = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            padded, pad_h, pad_w = pad_to_multiple(images, patch_size)

            outputs = model_instance(padded)

            outputs = unpad(outputs, pad_h, pad_w)

            loss = loss_fn(outputs, masks)
            tp, fp, fn, tn = compute_iou(outputs, masks)

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