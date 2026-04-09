import torch
import segmentation_models_pytorch as smp
from landcover.utils.util import compute_iou, pad_image, unpad_image

def test(model_instance, data_loader, loss_fn, patch_size=256, device="cpu", conf_matrix=None):
    model_instance.eval()
    total_tp = total_fp = total_fn = total_tn = None
    running_loss = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            padded, pad_h, pad_w = pad_image(images, patch_size)
            _, _, H_pad, W_pad = padded.shape

            logits_full = torch.zeros(
                (images.shape[0], model_instance.out_classes, H_pad, W_pad),
                device=device
            )
            count_map = torch.zeros((images.shape[0], 1, H_pad, W_pad), device=device)

            stride = patch_size // 2

            for i in range(0, H_pad - patch_size + 1, stride):
                for j in range(0, W_pad - patch_size + 1, stride):
                    patch = padded[:, :, i:i + patch_size, j:j + patch_size]
                    outputs = model_instance(patch)

                    logits_full[:, :, i:i + patch_size, j:j + patch_size] += outputs
                    count_map[:, :, i:i + patch_size, j:j + patch_size] += 1

            logits_full = logits_full / count_map
            logits_full = unpad_image(logits_full, pad_h, pad_w)

            loss = loss_fn(logits_full, masks)
            running_loss += loss.item()

            pred_mask = torch.argmax(logits_full, dim=1)
            tp, fp, fn, tn = compute_iou(logits_full, masks, model_instance.out_classes)

            if conf_matrix is not None:
                valid = masks != 255
                conf_matrix.update(pred_mask[valid], masks[valid])

            if total_tp is None:
                total_tp, total_fp, total_fn, total_tn = tp, fp, fn, tn
            else:
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

        avg_loss = running_loss / len(data_loader)
        m_iou = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn, reduction="macro")

    return avg_loss, m_iou.item()