import torch
from monai.inferers import sliding_window_inference
from landcover.utils.util import compute_stats, compute_metrics_from_stats

def test(model_instance, data_loader, loss_fn, patch_size=256, device="cpu", conf_matrix=None):
    model_instance.eval()

    running_loss = 0
    total_tp = total_fp = total_fn = total_tn = None

    with torch.no_grad():
        for images, masks, _ in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits_full = sliding_window_inference(
                inputs=images,
                roi_size=(patch_size, patch_size),
                sw_batch_size=4,
                predictor=model_instance,
                overlap=0.5
            )

            loss = loss_fn(logits_full, masks)
            tp, fp, fn, tn = compute_stats(logits_full, masks, model_instance.out_classes)

            if conf_matrix is not None:
                pred_mask = torch.argmax(logits_full, dim=1)
                valid = masks != 255
                conf_matrix.update(pred_mask[valid], masks[valid])

            running_loss += loss.item()

            if total_tp is None:
                total_tp, total_fp, total_fn, total_tn = tp, fp, fn, tn
            else:
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn



        avg_loss = running_loss / len(data_loader)
        metrics = compute_metrics_from_stats(total_tp, total_fp, total_fn, total_tn)
        metrics["avg_loss"] = avg_loss

    return metrics