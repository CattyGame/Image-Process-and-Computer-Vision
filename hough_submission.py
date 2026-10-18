import numpy as np

def hough_accumulate(edges, theta_range, rho_range, n_theta, n_rho,
                     grad_dir=None, delta_deg=None):
    """Bỏ phiếu vào mảng tích lũy. Nếu grad_dir khác None, chỉ bỏ phiếu
    trong dải +/- delta_deg quanh hướng gradient.
    edges: mảng (H, W) uint8 cạnh nhị phân.
    Trả về (A, thetas, rhos) với A là mảng (n_rho, n_theta) int32."""
    assert edges.ndim == 2, "edges phải là ảnh 2 chiều"
    thetas = np.linspace(theta_range[0], theta_range[1], n_theta, endpoint=False)
    rhos = np.linspace(rho_range[0], rho_range[1], n_rho, endpoint=True)
    A = np.zeros((n_rho, n_theta), dtype=np.int32)

    y_pts, x_pts = np.where(edges > 0)
    if len(y_pts) == 0:
        return A, thetas, rhos

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    d_rho = (rho_range[1] - rho_range[0]) / (n_rho - 1) if n_rho > 1 else 1.0
    rho_min = rho_range[0]

    if grad_dir is None or delta_deg is None:
        # Bỏ phiếu chuẩn tắc: quét qua từng cột góc theta
        for t_idx in range(n_theta):
            r_vals = x_pts * cos_t[t_idx] + y_pts * sin_t[t_idx]
            r_idx = np.round((r_vals - rho_min) / d_rho).astype(int)
            valid = (r_idx >= 0) & (r_idx < n_rho)
            np.add.at(A, (r_idx[valid], t_idx), 1)
    else:
        # Bỏ phiếu có định hướng gradient: lọc sai lệch góc tuần hoàn mod pi
        delta_rad = np.deg2rad(delta_deg)
        for y, x in zip(y_pts, x_pts):
            theta_g = grad_dir[y, x]
            diff = np.abs(thetas - theta_g) % np.pi
            dist = np.minimum(diff, np.pi - diff)
            valid_t = np.where(dist <= delta_rad)[0]
            if len(valid_t) == 0:
                continue
            r_vals = x * cos_t[valid_t] + y * sin_t[valid_t]
            r_idx = np.round((r_vals - rho_min) / d_rho).astype(int)
            valid_r = (r_idx >= 0) & (r_idx < n_rho)
            A[r_idx[valid_r], valid_t[valid_r]] += 1

    return A, thetas, rhos

def find_peaks(A, n_peaks, nms_radius=3, smooth_sigma=1.0, thetas=None, rhos=None):
    """Làm mượt A bằng nhân Gauss 2D đệm quấn vòng Mobius, triệt phi cực đại NMS 2D,
    trả về danh sách (rho, theta, votes) sắp xếp giảm dần."""
    n_rho, n_theta = A.shape

    # 1. Làm mượt bằng nhân Gauss 2D và đệm quấn vòng Mobius (rho -> -rho)
    if smooth_sigma > 0:
        rad = int(np.ceil(3 * smooth_sigma))
        ksize = 2 * rad + 1
        y, x = np.ogrid[-rad:rad+1, -rad:rad+1]
        kernel = np.exp(-(x**2 + y**2) / (2.0 * smooth_sigma**2))
        kernel /= kernel.sum()

        left_pad = np.flipud(A[:, -rad:]) if rad > 0 else np.empty((n_rho, 0))
        right_pad = np.flipud(A[:, :rad]) if rad > 0 else np.empty((n_rho, 0))
        padded_theta = np.hstack([left_pad, A, right_pad])
        padded = np.pad(padded_theta, ((rad, rad), (0, 0)), mode='constant')

        A_smooth = np.zeros_like(A, dtype=np.float64)
        for i in range(ksize):
            for j in range(ksize):
                A_smooth += kernel[i, j] * padded[i:i + n_rho, j:j + n_theta]
    else:
        A_smooth = A.astype(np.float64)

    # 2. Triệt phi cực đại NMS 2D với đệm quấn vòng Mobius
    left_nms = np.flipud(A_smooth[:, -nms_radius:]) if nms_radius > 0 else np.empty((n_rho, 0))
    right_nms = np.flipud(A_smooth[:, :nms_radius]) if nms_radius > 0 else np.empty((n_rho, 0))
    padded_nms_theta = np.hstack([left_nms, A_smooth, right_nms])
    padded_nms = np.pad(padded_nms_theta, ((nms_radius, nms_radius), (0, 0)), mode='constant')

    peaks = []
    for r in range(n_rho):
        r_pad = r + nms_radius
        for t in range(n_theta):
            t_pad = t + nms_radius
            val = A_smooth[r, t]
            if val <= 0:
                continue
            window = padded_nms[r_pad - nms_radius:r_pad + nms_radius + 1,
                                t_pad - nms_radius:t_pad + nms_radius + 1]
            if val >= np.max(window):
                r_val = rhos[r] if rhos is not None else float(r)
                t_val = thetas[t] if thetas is not None else float(t)
                peaks.append((r_val, t_val, int(A[r, t]), val))

    peaks.sort(key=lambda p: p[3], reverse=True)
    return [(p[0], p[1], p[2]) for p in peaks[:n_peaks]]

def hough_multiscale(edges, grad_dir=None, delta_deg=None, n_levels=2, refine_factor=8,
                     n_peaks=5, d_theta_coarse_deg=1.0, d_rho_coarse=2.0,
                     nms_radius=3, smooth_sigma=1.0):
    """Hough thô-đến-mịn (Coarse-to-fine).
    edges: ảnh cạnh nhị phân (H, W) uint8.
    Trả về danh sách (rho, theta) đã tinh chỉnh độ chính xác cao."""
    assert edges.ndim == 2, "edges phải là ảnh 2 chiều"
    H, W = edges.shape
    diag = np.sqrt(H**2 + W**2)

    # 1. Tầng thô (Coarse level)
    d_theta_coarse = np.deg2rad(d_theta_coarse_deg)
    n_theta_coarse = int(np.ceil(np.pi / d_theta_coarse))
    n_rho_coarse = int(np.ceil(2 * diag / d_rho_coarse)) + 1

    A_c, t_c, r_c = hough_accumulate(
        edges, (0.0, np.pi), (-diag, diag),
        n_theta_coarse, n_rho_coarse,
        grad_dir=grad_dir, delta_deg=delta_deg
    )
    coarse_peaks = find_peaks(
        A_c, n_peaks=n_peaks, nms_radius=nms_radius,
        smooth_sigma=smooth_sigma, thetas=t_c, rhos=r_c
    )

    y_pts, x_pts = np.where(edges > 0)
    if len(y_pts) == 0:
        return []

    refined_lines = []

    # 2. Tầng mịn cục bộ (Fine level)
    for rho_star, theta_star, _ in coarse_peaks:
        cur_rho, cur_theta = rho_star, theta_star
        cur_d_theta, cur_d_rho = d_theta_coarse, d_rho_coarse

        # Lọc inliers trong lân cận đường thẳng: d_max = (D/2)*Delta_theta + Delta_rho
        rho_span = (diag / 2.0) * cur_d_theta + cur_d_rho
        dist_to_line = np.abs(x_pts * np.cos(cur_theta) + y_pts * np.sin(cur_theta) - cur_rho)
        inlier_mask = dist_to_line <= (rho_span * 1.2)

        if np.sum(inlier_mask) < 3:
            t_norm = cur_theta % (2 * np.pi)
            if t_norm >= np.pi:
                t_norm -= np.pi
                cur_rho = -cur_rho
            refined_lines.append((cur_rho, t_norm))
            continue

        x_sub, y_sub = x_pts[inlier_mask], y_pts[inlier_mask]

        for level in range(1, n_levels):
            n_steps = refine_factor * 2 + 1
            d_rho_span = (diag / 2.0) * cur_d_theta + cur_d_rho
            d_theta_span = cur_d_theta

            step_theta = (2.0 * d_theta_span) / (n_steps - 1)
            step_rho = (2.0 * d_rho_span) / (n_steps - 1)

            local_thetas = np.linspace(cur_theta - d_theta_span, cur_theta + d_theta_span, n_steps)
            local_rhos = np.linspace(cur_rho - d_rho_span, cur_rho + d_rho_span, n_steps)

            A_fine = np.zeros((n_steps, n_steps), dtype=np.int32)
            for t_idx, t_val in enumerate(local_thetas):
                r_vals = x_sub * np.cos(t_val) + y_sub * np.sin(t_val)
                r_idx = np.round((r_vals - local_rhos[0]) / step_rho).astype(int)
                valid = (r_idx >= 0) & (r_idx < n_steps)
                np.add.at(A_fine, (r_idx[valid], t_idx), 1)

            best_r, best_t = np.unravel_index(np.argmax(A_fine), A_fine.shape)
            cur_theta = local_thetas[best_t]
            cur_rho = local_rhos[best_r]
            cur_d_theta = step_theta
            cur_d_rho = step_rho

        # Chuẩn hóa canonical theta thuộc [0, pi)
        final_theta = cur_theta % (2 * np.pi)
        final_rho = cur_rho
        if final_theta >= np.pi:
            final_theta -= np.pi
            final_rho = -final_rho
        refined_lines.append((final_rho, final_theta))

    return refined_lines
