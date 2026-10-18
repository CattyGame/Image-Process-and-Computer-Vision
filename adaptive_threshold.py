"""Ky thuat 2 - Nguong thich nghi bang anh tich phan.

Tu viet theo dac ta muc 3.5 trong tai lieu bo tro (mon Thi giac may tinh 121036).
Chi dung numpy (+ time, matplotlib o script kiem thu).

Cong thuc ap dung:
  (6)  S(x,y) = I(x,y) + S(x-1,y) + S(x,y-1) - S(x-1,y-1)
  (7)  Sigma = S(x2,y2) - S(x1-1,y2) - S(x2,y1-1) + S(x1-1,y1-1)
  (8)  Bradley-Roth: T(x,y) = m(x,y)*(1-t), t ~ 0.15
  (9)  Sauvola: T = m*[1 + k*(s/R - 1)], k ~ 0.34, R = 128
  (10) s^2 = Sigma2/w^2 - (Sigma1/w^2)^2

Quy uoc chi so: anh numpy I[y, x] (hang truoc, cot sau).
"""

import numpy as np


def integral_image(img):
    """Tra ve S kich thuoc (H+1, W+1), dtype int64, hang/cot 0 bang 0.

    Ep kieu sang int64 TRUOC khi cumsum de tranh tran int32
    (anh 2000x3000: tong toi da ~1.5e9 vuot int32).
    """
    img = np.asarray(img)
    H, W = img.shape
    S = np.zeros((H + 1, W + 1), dtype=np.int64)
    # cumsum 2 lan = truy hoi (6), tuong duong mot luot quet O(HW)
    S[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.int64, copy=False),
                                    axis=0), axis=1)
    return S


def box_sum(S, y1, x1, y2, x2):
    """Tong vung [y1..y2, x1..x2] (bao gom ca bien), vector hoa duoc.

    S la bang tich phan da dem (H+1, W+1); y1/x1/y2/x2 co the la so
    hoac mang numpy (broadcast voi nhau). Cong thuc (7) tren chi so dem:
    Sigma = S[y2+1,x2+1] - S[y1,x2+1] - S[y2+1,x1] + S[y1,x1].
    """
    y1 = np.asarray(y1)
    x1 = np.asarray(x1)
    y2 = np.asarray(y2)
    x2 = np.asarray(x2)
    return (S[y2 + 1, x2 + 1] - S[y1, x2 + 1]
            - S[y2 + 1, x1] + S[y1, x1])


def local_mean_std(img, radius):
    """Tra ve (mean, std), moi cai (H, W) float64. Dung 2 anh tich phan.

    S1 cua I, S2 cua I^2; cong thuc (10). Kep var >= 0 truoc sqrt de
    tranh nan do sai so triet tieu tren anh gan dong nhat.
    Cua so w = 2*radius+1, kep bien bang clamp chi so.
    """
    img = np.asarray(img)
    H, W = img.shape
    f = img.astype(np.float64, copy=False)

    S1 = np.zeros((H + 1, W + 1), dtype=np.float64)
    S1[1:, 1:] = np.cumsum(np.cumsum(f, axis=0), axis=1)
    S2 = np.zeros((H + 1, W + 1), dtype=np.float64)
    S2[1:, 1:] = np.cumsum(np.cumsum(f * f, axis=0), axis=1)

    yy, xx = np.mgrid[0:H, 0:W]
    y1 = np.clip(yy - radius, 0, H - 1)
    y2 = np.clip(yy + radius, 0, H - 1)
    x1 = np.clip(xx - radius, 0, W - 1)
    x2 = np.clip(xx + radius, 0, W - 1)
    area = ((y2 - y1 + 1) * (x2 - x1 + 1)).astype(np.float64)

    sum1 = box_sum(S1, y1, x1, y2, x2)
    sum2 = box_sum(S2, y1, x1, y2, x2)
    mean = sum1 / area
    var = sum2 / area - mean * mean
    var = np.maximum(var, 0.0)
    return mean, np.sqrt(var)


def adaptive_threshold(img, radius=15, method='bradley', t=0.15, k=0.34):
    """Tra ve anh nhi phan (H, W) uint8 voi gia tri 0 / 255.

    method='bradley': T = m*(1-t) (cong thuc 8).
    method='sauvola': T = m*[1 + k*(s/128 - 1)] (cong thuc 9, R = 128).
    Diem anh la foreground (255) neu I < T.
    """
    img = np.asarray(img)
    if method not in ('bradley', 'sauvola'):
        raise ValueError("method phai la 'bradley' hoac 'sauvola'")
    mean, std = local_mean_std(img, radius)
    f = img.astype(np.float64, copy=False)
    if method == 'bradley':
        T = mean * (1.0 - t)
    else:
        T = mean * (1.0 + k * (std / 128.0 - 1.0))
    return np.where(f < T, 255, 0).astype(np.uint8)
