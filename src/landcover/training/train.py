import segmentation_models_pytorch as smp
import torch


def train(model_instance, data_loader, opt, loss_fn, device="cpu"):
    model_instance.train()
    running_loss = 0
    running_iou = 0

    for images, masks in data_loader:
        images = images.to(device)
        masks = masks.to(device)

        opt.zero_grad()

        outputs = model_instance(images)

        loss = loss_fn(outputs, masks)
        iou = metric_out(outputs, masks)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 1.0)
        opt.step()

        running_loss += loss.item()
        running_iou += iou

    avg_loss = running_loss / len(data_loader)
    avg_iou = running_iou / len(data_loader)

    return avg_loss, avg_iou

def test(model_instance, data_loader, loss_fn, device="cpu"):
    model_instance.eval()
    running_loss = 0
    running_iou = 0
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model_instance(images)

            loss = loss_fn(outputs, masks)
            iou = metric_out(outputs, masks)

            running_loss += loss.item()
            running_iou += iou

    avg_loss = running_loss / len(data_loader)
    avg_iou = running_iou / len(data_loader)

    return avg_loss, avg_iou


def metric_out(outputs, masks):
    preds = torch.argmax(outputs, dim=1).long()
    masks = masks.long()

    tp, fp, fn, tn = smp.metrics.get_stats(
        preds,
        masks,
        mode='multiclass',
        num_classes=9,
    )

    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro')

    return iou


