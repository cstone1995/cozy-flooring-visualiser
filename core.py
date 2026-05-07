"""
core.py - Cozy Flooring Visualiser Core Logic

Free/self-hosted AI flooring visualiser engine:
- SegFormer ADE20K floor detection
- Stronger mask cleanup
- Better texture tiling/scaling
- Perspective warp into the detected floor plane
- Lighting/shadow preservation
- Feathered edges so it looks less pasted
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ADE20K class indices after reduce_labels.
# Floor original class is 4; after reduce_labels = 3.
FLOOR_CLASS_IDS = [3]

_model = None
_processor = None
_device = None


def load_model(device: Optional[str] = None):
    """Load SegFormer once and cache it."""
    global _model, _processor, _device

    if _model is not None:
        return _model, _processor, _device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading SegFormer model on {device}...")
    model_name = "nvidia/segformer-b5-finetuned-ade-640-640"

    _processor = SegformerImageProcessor.from_pretrained(model_name)
    _model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    _model.to(device)
    _model.eval()
    _device = device

    print("SegFormer model loaded successfully!")
    return _model, _processor, _device


def get_floor_mask(
    image: np.ndarray,
    model=None,
    processor=None,
    device: Optional[str] = None,
    mask_dilate: int = 2,
) -> np.ndarray:
    """Detect floor using SegFormer and return clean binary mask."""
    global _model, _processor, _device

    if model is None:
        if _model is None:
            load_model()
        model, processor, device = _model, _processor, _device

    h, w = image.shape[:2]
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )
    predictions = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

    mask = np.zeros((h, w), dtype=np.uint8)
    for class_id in FLOOR_CLASS_IDS:
        mask = np.logical_or(mask, predictions == class_id).astype(np.uint8)

    # Clean and join floor areas.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = _keep_largest_component(mask)

    if mask.sum() < (h * w * 0.035):
        mask = _expand_floor_mask(image, mask, predictions)

    # Very small dilation helps hide tiny missed edges at skirting boards.
    if mask_dilate > 0:
        k = max(1, int(mask_dilate))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask.astype(np.uint8)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


def _expand_floor_mask(image: np.ndarray, mask: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]

    if mask.sum() > 0:
        floor_pixels = image[mask > 0]
        reference_color = np.median(floor_pixels, axis=0)
    else:
        sample_region = image[int(h * 0.72): int(h * 0.96), int(w * 0.2): int(w * 0.8)]
        reference_color = np.median(sample_region.reshape(-1, 3), axis=0)

    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(np.uint8([[reference_color]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    diff = np.sqrt(np.sum((image_lab - ref_lab) ** 2, axis=2))

    bottom_region = predictions[int(h * 0.55):, :]
    unique, counts = np.unique(bottom_region, return_counts=True)
    if len(unique) == 0:
        return mask

    dominant_class = unique[np.argmax(counts)]
    class_mask = (predictions == dominant_class).astype(np.uint8)

    color_threshold = np.percentile(diff, 28)
    color_mask = (diff < color_threshold).astype(np.uint8)
    color_mask[: int(h * 0.28), :] = 0

    combined = np.logical_or(class_mask, color_mask).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    return _keep_largest_component(combined)


def get_four_corners_from_mask(mask: np.ndarray) -> np.ndarray:
    """Estimate the visible floor plane corners from the mask contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = mask.shape[:2]
        return np.array([[0, h * 0.6], [w, h * 0.6], [w, h], [0, h]], dtype=np.float32)

    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour).reshape(-1, 2)

    # Extreme points generally work better for floor plane than approxPolyDP.
    tl = hull[np.argmin(hull[:, 0] + hull[:, 1])]
    tr = hull[np.argmax(hull[:, 0] - hull[:, 1])]
    br = hull[np.argmax(hull[:, 0] + hull[:, 1])]
    bl = hull[np.argmin(hull[:, 0] - hull[:, 1])]

    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    return order_points(corners)


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def prepare_texture(texture: np.ndarray, tile_size: int = 90) -> np.ndarray:
    """Crop to square-ish texture and resize to a clean tile size."""
    if texture.ndim == 2:
        texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2RGB)
    if texture.shape[2] == 4:
        texture = texture[:, :, :3]

    h, w = texture.shape[:2]
    # Centre crop to square to avoid stretched product thumbnails.
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    texture = texture[y0: y0 + side, x0: x0 + side]

    tile_size = int(np.clip(tile_size, 35, 260))
    texture = cv2.resize(texture, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    return texture


def tile_texture(texture: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    th, tw = texture.shape[:2]
    target_h, target_w = target_size
    tiles_y = (target_h // th) + 4
    tiles_x = (target_w // tw) + 4
    tiled = np.tile(texture, (tiles_y, tiles_x, 1))
    return tiled[:target_h, :target_w]


def apply_perspective_transform(
    texture_img: np.ndarray,
    target_shape: Tuple[int, int],
    mask: np.ndarray,
    tile_size: int = 90,
) -> np.ndarray:
    """Create a tiled texture plane and warp it into the detected floor shape."""
    h, w = target_shape
    texture_tile = prepare_texture(texture_img, tile_size=tile_size)

    corners = get_four_corners_from_mask(mask)

    # Estimate a source plane big enough for the floor; this avoids using one tiny
    # product thumbnail as the whole floor.
    top_width = np.linalg.norm(corners[1] - corners[0])
    bottom_width = np.linalg.norm(corners[2] - corners[3])
    left_height = np.linalg.norm(corners[3] - corners[0])
    right_height = np.linalg.norm(corners[2] - corners[1])

    plane_w = max(300, int(max(top_width, bottom_width) * 1.4))
    plane_h = max(300, int(max(left_height, right_height) * 1.4))

    tiled_plane = tile_texture(texture_tile, (plane_h, plane_w))

    src_pts = np.array(
        [[0, 0], [plane_w - 1, 0], [plane_w - 1, plane_h - 1], [0, plane_h - 1]],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(src_pts, corners.astype(np.float32))
    warped = cv2.warpPerspective(
        tiled_plane,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped


def blend_with_lighting(
    original_room: np.ndarray,
    warped_texture: np.ndarray,
    mask: np.ndarray,
    blend_strength: float = 0.62,
    edge_feather: int = 17,
) -> np.ndarray:
    """Blend new floor while preserving original shadows/highlights."""
    blend_strength = float(np.clip(blend_strength, 0.0, 1.0))

    room_lab = cv2.cvtColor(original_room, cv2.COLOR_RGB2LAB).astype(np.float32)
    tex_lab = cv2.cvtColor(warped_texture, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Preserve floor luminance/shadows from original, but keep enough texture colour.
    mixed_lab = tex_lab.copy()
    mixed_lab[:, :, 0] = blend_strength * room_lab[:, :, 0] + (1 - blend_strength) * tex_lab[:, :, 0]

    # Slightly soften chroma so aggressive product thumbnails look less noisy.
    mixed_lab[:, :, 1] = 0.92 * mixed_lab[:, :, 1] + 0.08 * room_lab[:, :, 1]
    mixed_lab[:, :, 2] = 0.92 * mixed_lab[:, :, 2] + 0.08 * room_lab[:, :, 2]

    replacement = cv2.cvtColor(np.clip(mixed_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    alpha = feather_mask(mask, edge_feather)
    alpha_3 = np.dstack([alpha, alpha, alpha])
    result = (alpha_3 * replacement + (1 - alpha_3) * original_room).astype(np.uint8)
    return result


def feather_mask(mask: np.ndarray, feather_amount: int = 17) -> np.ndarray:
    """Return smooth 0..1 alpha mask with feathered edges."""
    feather_amount = int(max(1, feather_amount))
    if feather_amount % 2 == 0:
        feather_amount += 1

    alpha = mask.astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (feather_amount, feather_amount), 0)
    alpha = np.clip(alpha, 0, 1)
    return alpha


def process_room(
    room_image: np.ndarray,
    texture_image: np.ndarray,
    model=None,
    processor=None,
    device: Optional[str] = None,
    blend_strength: float = 0.62,
    tile_size: int = 90,
    edge_feather: int = 17,
    mask_dilate: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full pipeline: detect floor, warp texture, blend result."""
    mask = get_floor_mask(room_image, model, processor, device, mask_dilate=mask_dilate)
    warped = apply_perspective_transform(texture_image, room_image.shape[:2], mask, tile_size=tile_size)
    result = blend_with_lighting(
        room_image,
        warped,
        mask,
        blend_strength=blend_strength,
        edge_feather=edge_feather,
    )
    return result, mask
