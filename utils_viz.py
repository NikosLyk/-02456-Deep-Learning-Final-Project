import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def _to_numpy_img(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 4 and x.shape[1] == 1:
            return [x[i,0].numpy() for i in range(x.shape[0])]
        if x.ndim == 3 and x.shape[0] == 1:
            return x[0].numpy()
    return x

def _to01(a):
    a = np.asarray(a, dtype=np.float32)
    if a.max() > 1.5:  # handle 0..255
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)

def overlay_mask(img, mask, alpha=0.35):
    img  = _to01(img)
    mask = (np.asarray(mask) > 0.5).astype(np.float32)
    rgb  = np.stack([img, img, img], axis=-1)
    tint = np.zeros_like(rgb); tint[..., 0] = mask  # red
    out  = (1 - alpha) * rgb + alpha * tint
    return np.clip(out, 0.0, 1.0)

def show_augmented_labeled(batch, n=2):
    xw = batch["x_w"].cpu().numpy()  # [B,1,H,W]
    xs = batch["x_s"].cpu().numpy()
    y  = batch["y"].cpu().numpy()

    fig, axes = plt.subplots(n, 4, figsize=(16, 8))
    fig.suptitle("Labeled augmentations (weak vs strong + mask)")

    for i in range(n):
        iw = _to01(xw[i,0])
        is_ = _to01(xs[i,0])
        m  = (y[i,0] > 0.5).astype(np.float32)

        axes[i,0].imshow(iw, cmap="gray", vmin=0, vmax=1); axes[i,0].set_title("weak");  axes[i,0].axis("off")
        axes[i,1].imshow(is_, cmap="gray", vmin=0, vmax=1); axes[i,1].set_title("strong");axes[i,1].axis("off")
        axes[i,2].imshow(overlay_mask(is_, m), vmin=0, vmax=1); axes[i,2].set_title("strong + mask"); axes[i,2].axis("off")
        axes[i,3].imshow(overlay_mask(iw, m),  vmin=0, vmax=1);     axes[i,3].set_title("weak + mask");   axes[i,3].axis("off")

    plt.tight_layout(); plt.show()

def show_augmented_unlabeled(batch, n=2):
    xw = batch["x_w"].cpu().numpy()
    xs = batch["x_s"].cpu().numpy()

    fig, axes = plt.subplots(n, 2, figsize=(9, 9))
    fig.suptitle("Unlabeled augmentations (weak vs strong)")
    for i in range(n):
        iw = _to01(xw[i,0])
        is_ = _to01(xs[i,0])

        axes[i,0].imshow(iw, cmap="gray", vmin=0, vmax=1); axes[i,0].set_title("weak");  axes[i,0].axis("off")
        axes[i,1].imshow(is_, cmap="gray", vmin=0, vmax=1); axes[i,1].set_title("strong");axes[i,1].axis("off")

    plt.tight_layout(); plt.show()


def save_pred_grid(x, y, logits, out_path, thresh=0.5):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imgs = _to_numpy_img(x); gts = _to_numpy_img(y)
    preds = (torch.sigmoid(logits).detach().cpu().numpy() > thresh).astype(np.float32)
    preds = [preds[i,0] for i in range(preds.shape[0])]
    n = min(6, len(imgs))
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2*n))
    if n == 1: axes = np.array([axes])
    for i in range(n):
        axes[i,0].imshow(imgs[i], cmap="gray"); axes[i,0].set_title("image")
        axes[i,1].imshow(gts[i], cmap="gray");  axes[i,1].set_title("GT mask")
        axes[i,2].imshow(preds[i], cmap="gray");axes[i,2].set_title("Pred mask")
        for j in range(3): axes[i,j].axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close(fig)
