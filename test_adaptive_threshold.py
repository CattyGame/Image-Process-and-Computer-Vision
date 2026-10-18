"""Kiem thu Nguoi 2: so sanh voi vong lap naive + anh tong hop."""

import time
import numpy as np

from adaptive_threshold import (
    integral_image, box_sum, local_mean_std, adaptive_threshold,
)


def naive_box(img, y1, x1, y2, x2):
    return img[y1:y2 + 1, x1:x2 + 1].sum()


def naive_mean_std(img, radius):
    H, W = img.shape
    mean = np.zeros((H, W))
    std = np.zeros((H, W))
    for y in range(H):
        for x in range(W):
            win = img[max(0, y - radius):y + radius + 1,
                      max(0, x - radius):x + radius + 1].astype(float)
            mean[y, x] = win.mean()
            std[y, x] = win.std()
    return mean, std


rng = np.random.default_rng(0)

# 1. box_sum khop naive
img = rng.integers(0, 256, size=(30, 40)).astype(np.uint8)
S = integral_image(img)
assert S.shape == (31, 41) and S.dtype == np.int64
assert (S[0, :] == 0).all() and (S[:, 0] == 0).all()
for _ in range(200):
    y1, y2 = sorted(rng.integers(0, 30, size=2))
    x1, x2 = sorted(rng.integers(0, 40, size=2))
    assert box_sum(S, y1, x1, y2, x2) == naive_box(img, y1, x1, y2, x2)
# vector hoa
ys = np.array([0, 5, 10]); xs = np.array([0, 3, 7])
r = box_sum(S, ys[:, None], xs, ys[:, None] + 4, xs + 6)
assert r.shape == (3, 3)
print("1. box_sum khop naive: OK")

# 2. mean/std khop naive
mean, std = local_mean_std(img, 3)
m0, s0 = naive_mean_std(img, 3)
assert np.abs(mean - m0).max() < 1e-9, np.abs(mean - m0).max()
assert np.abs(std - s0).max() < 1e-9, np.abs(std - s0).max()
assert not np.isnan(std).any()
print("2. local_mean_std khop naive, khong nan: OK")

# 3. tran so: anh 2000x3000 gia tri 255 (tong ~1.5e9 vuot int32)
big = np.full((200, 300), 255, np.uint8)
Sb = integral_image(big)
assert Sb[-1, -1] == 200 * 300 * 255
# anh phang: var phai = 0, khong nan
_, std_flat = local_mean_std(np.full((20, 20), 128, np.uint8), 5)
assert (std_flat == 0).all()
print("3. tran so + anh phang var=0: OK")

# 4. anh tong hop: nen gradient + chu toi + vung nen phang
H, W = 120, 160
Y, X = np.mgrid[0:H, 0:W]
bg = (60 + 120 * X / W).astype(np.uint8)  # sang dan tu trai sang phai
syn = bg.copy()
syn[20:45, 20:140] = np.minimum(syn[20:45, 20:140], 30)   # vach chu toi
syn[70:95, 40:120] = np.minimum(syn[70:95, 40:120], 40)
flat = syn[100:115, 120:155].copy()  # vung nen phang giu lai de dem dom
syn = syn.astype(np.int16)
syn[100:115, 120:155] += np.round(rng.normal(0, 10, size=(15, 35))).astype(np.int16)  # nhieu Gauss tren nen phang
syn = np.clip(syn, 0, 255).astype(np.uint8)
out_b = adaptive_threshold(syn, radius=15, method='bradley', t=0.15)
out_s = adaptive_threshold(syn, radius=15, method='sauvola', k=0.34)
speck_b = (out_b[100:115, 120:155] == 255).sum()
speck_s = (out_s[100:115, 120:155] == 255).sum()
print(f"4. dom tren nen phang: Bradley={speck_b}, Sauvola={speck_s}")
assert speck_s <= speck_b, "Sauvola phai sach dom hon Bradley"
assert set(np.unique(out_b)) <= {0, 255} and set(np.unique(out_s)) <= {0, 255}
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(syn, cmap="gray"); ax[0].set_title("anh tong hop")
    ax[1].imshow(out_b, cmap="gray"); ax[1].set_title("Bradley-Roth")
    ax[2].imshow(out_s, cmap="gray"); ax[2].set_title("Sauvola")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig("vi_du_nguoi2.png", dpi=100)
    print("   da luu vi_du_nguoi2.png")
except ImportError:
    print("   (bo qua ve hinh: chua co matplotlib)")

# 5. do thoi gian local_mean_std theo radius (cho Nguoi 4, muc 3.6c)
timg = rng.integers(0, 256, size=(200, 200)).astype(np.uint8)
for r in [1, 2, 4, 8, 16, 32, 64]:
    t0 = time.perf_counter()
    local_mean_std(timg, r)
    print(f"   radius={r:3d}: {time.perf_counter() - t0:.4f}s")

print("TAT CA KIEM TRA DAT")
