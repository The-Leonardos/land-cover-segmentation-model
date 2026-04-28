import torch
from landcover.utils.util import compute_stats, compute_metrics_from_stats

def train(model_instance, data_loader, opt, loss_fn, device="cpu", conf_matrix=None):
    model_instance.train()
    total_tp = total_fp = total_fn = total_tn = None
    running_loss = 0

    total_patches = 0
    total_rejections = 0

    for images, masks, rejected_patches in data_loader:
        images = images.to(device)
        masks = masks.to(device)

        batch_size = images.size(0)
        total_patches += batch_size
        total_rejections += rejected_patches.sum().item()

        opt.zero_grad()

        outputs = model_instance(images)

        loss = loss_fn(outputs, masks)
        tp, fp, fn, tn = compute_stats(outputs, masks, model_instance.out_classes)

        if conf_matrix is not None:
            preds = torch.argmax(outputs, dim=1)
            valid = masks != 255
            conf_matrix.update(preds[valid], masks[valid])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 1.0)
        opt.step()

        running_loss += loss.item()

        tp = tp.sum(dim=0)
        fp = fp.sum(dim=0)
        fn = fn.sum(dim=0)
        tn = tn.sum(dim=0)

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
    metrics = compute_metrics_from_stats(total_tp, total_fp, total_fn, total_tn)
    metrics["avg_loss"] = avg_loss

    metrics["total_patches"] = total_patches
    metrics["total_rejections"] = total_rejections

    return metrics