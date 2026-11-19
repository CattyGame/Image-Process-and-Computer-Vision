import numpy as np
from collections import deque


def region_growing_classic(img, seeds, T, order="stack"):
    """Grow a region from each seed using four-connected neighbours.

    Unassigned pixels have label -1. Seeds are labelled before expansion
    so a region cannot claim another region's starting pixel.
    """
    image = np.asarray(img, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("img phải là ảnh xám 2 chiều")
    if not np.all(np.isfinite(image)):
        raise ValueError("img phải chứa giá trị hữu hạn")
    if order not in ("stack", "queue"):
        raise ValueError("order phải là 'stack' hoặc 'queue'")
    if not np.isfinite(T) or T < 0:
        raise ValueError("T phải là số hữu hạn không âm")

    height, width = image.shape
    labels = np.full(image.shape, -1, dtype=np.int32)
    seed_list = []

    for seed in seeds:
        if len(seed) != 2:
            raise ValueError("Mỗi seed phải có dạng (row, col)")
        row, col = seed
        if not isinstance(row, (int, np.integer)):
            raise ValueError("Tọa độ row phải là số nguyên")
        if not isinstance(col, (int, np.integer)):
            raise ValueError("Tọa độ col phải là số nguyên")
        row, col = int(row), int(col)
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError("Seed nằm ngoài kích thước ảnh")
        if (row, col) in seed_list:
            raise ValueError("Danh sách seeds có tọa độ trùng")
        seed_list.append((row, col))

    for label_id, (row, col) in enumerate(seed_list):
        labels[row, col] = label_id

    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for label_id, (start_row, start_col) in enumerate(seed_list):
        pending = deque([(start_row, start_col)])
        region_total = float(image[start_row, start_col])
        region_size = 1

        while pending:
            row, col = pending.pop() if order == "stack" else pending.popleft()
            for dr, dc in offsets:
                next_row, next_col = row + dr, col + dc
                if not (0 <= next_row < height and 0 <= next_col < width):
                    continue
                if labels[next_row, next_col] != -1:
                    continue

                value = float(image[next_row, next_col])
                if abs(value - region_total / region_size) > T:
                    continue

                labels[next_row, next_col] = label_id
                region_total += value
                region_size += 1
                pending.append((next_row, next_col))

    return labels

