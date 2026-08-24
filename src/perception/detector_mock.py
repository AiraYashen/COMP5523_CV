from __future__ import annotations

import cv2
import numpy as np

from src.common.schemas import Detection


class MockDetector:
    def __init__(self) -> None:
        pass

    def detect(
        self, rgb_image: np.ndarray, prompt: str, threshold: float = 0.25
    ) -> list[Detection]:
        _ = threshold
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        lower_sat = 60
        lower_val = 40

        # Prompt-guided color heuristic for local runnable fallback.
        prompt_lower = prompt.lower()
        if any(token in prompt_lower for token in ["red", "coke", "cola", "soda"]):
            mask1 = cv2.inRange(
                hsv,
                np.array([0, lower_sat, lower_val], dtype=np.uint8),
                np.array([10, 255, 255], dtype=np.uint8),
            )
            mask2 = cv2.inRange(
                hsv,
                np.array([170, lower_sat, lower_val], dtype=np.uint8),
                np.array([180, 255, 255], dtype=np.uint8),
            )
            mask = cv2.bitwise_or(mask1, mask2)
        elif any(token in prompt_lower for token in ["green", "bottle"]):
            mask = cv2.inRange(
                hsv,
                np.array([35, lower_sat, lower_val], dtype=np.uint8),
                np.array([90, 255, 255], dtype=np.uint8),
            )
        else:
            sat = hsv[:, :, 1]
            mask = cv2.inRange(
                sat,
                np.array(50, dtype=np.uint8),
                np.array(255, dtype=np.uint8),
            )

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        image_h, image_w = rgb_image.shape[:2]
        image_area = float(image_h * image_w)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 400:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            score = min(0.99, max(0.3, area / image_area * 12.0))
            detections.append(
                Detection(
                    bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
                    label=f"mock:{prompt}",
                    score=score,
                )
            )

        if detections:
            detections.sort(key=lambda item: item.score, reverse=True)
            return detections

        # Fallback center box keeps the session runnable even when heuristic fails.
        center_w = image_w * 0.18
        center_h = image_h * 0.22
        cx = image_w / 2
        cy = image_h * 0.65
        return [
            Detection(
                bbox_xyxy=(
                    cx - center_w / 2,
                    cy - center_h / 2,
                    cx + center_w / 2,
                    cy + center_h / 2,
                ),
                label=f"mock:{prompt}",
                score=0.31,
            )
        ]
