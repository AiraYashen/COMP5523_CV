from __future__ import annotations

import cv2
import numpy as np


def render_camera_preview(
    rgb_image: np.ndarray,
    title: str = "Camera Preview",
    status_lines: list[str] | None = None,
) -> np.ndarray:
    frame = cv2.cvtColor(rgb_image.copy(), cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (18, 18, 18), -1)
    cv2.putText(
        frame,
        title,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    lines = status_lines or []
    y = 62
    for line in lines[:4]:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        y += 28
    return frame
