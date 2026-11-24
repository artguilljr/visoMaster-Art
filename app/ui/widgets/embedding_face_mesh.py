"""Face texture fusion and projection for the Embedding Editor."""

from math import cos, radians, sin

import cv2
import numpy as np
from scipy.spatial import Delaunay, QhullError


def usable_mesh(photo):
    points = photo.landmarks
    return points is not None and points.shape == (478, 3) and np.isfinite(points).all()


def quality_flags(photo, others):
    flags = []
    if photo.sharpness < 45:
        flags.append("Low sharpness")
    if not usable_mesh(photo):
        flags.append("Dense landmarks unavailable")
    elif np.mean(np.any((photo.landmarks[:, :2] < 0) | (photo.landmarks[:, :2] >= 224), axis=1)) > 0.12:
        flags.append("Part of the face falls outside the crop")
    if any(
        other is not photo and not other.origin
        and np.mean(np.abs(photo.thumbnail - other.thumbnail)) < 3
        for other in others
    ):
        flags.append("Possible duplicate")
    return flags


def triangle_maps(source, target, triangles, width, height, weights=None):
    """Map each destination triangle back to its source texture coordinates."""
    map_x = np.full((height, width), -1, dtype=np.float32)
    map_y = np.full_like(map_x, -1)
    coverage = np.zeros_like(map_x)
    for index, triangle in enumerate(triangles):
        dst = np.asarray(target[triangle], dtype=np.float32)
        src = np.asarray(source[triangle], dtype=np.float32)
        area = cv2.contourArea(dst, oriented=True)
        source_area = cv2.contourArea(src, oriented=True)
        if abs(area) < 0.25 or area * source_area <= 0:
            continue
        x, y, w, h = cv2.boundingRect(dst)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        local = dst - np.array([x0, y0], dtype=np.float32)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(local * 16).astype(np.int32), 1, shift=4)
        ys, xs = np.nonzero(mask)
        inverse = cv2.getAffineTransform(local, src)
        yy, xx = ys + y0, xs + x0
        map_x[yy, xx] = inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]
        map_y[yy, xx] = inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]
        coverage[yy, xx] = 1 if weights is None else weights[index]
    valid = (map_x >= 0) & (map_y >= 0) & (map_x < 223) & (map_y < 223)
    return map_x, map_y, coverage * valid


def fuse_face_texture(references, front):
    """Anchor the face to a sharp reference and blend aligned, color-matched detail."""
    candidates = [photo for photo in references if usable_mesh(photo)]
    anchors = [photo for photo in front if usable_mesh(photo)] or candidates
    if not anchors:
        return max(front or references, key=lambda photo: photo.sharpness).rgb, None, None
    anchor = max(anchors, key=lambda photo: photo.sharpness)
    points = anchor.landmarks[:468].astype(np.float32)
    xy = points[:, :2]
    try:
        triangles = Delaunay(xy).simplices
    except QhullError:
        return anchor.rgb, None, None

    base = anchor.rgb.astype(np.float32)
    if len(candidates) == 1:
        return anchor.rgb, points, triangles
    # Protect eyes, nose and mouth from pose/expression changes in side photos.
    yy, xx = np.ogrid[:224, :224]
    center = xy[[33, 263, 1, 13, 14]].mean(axis=0)
    central = np.exp(-(((xx - center[0]) / 36) ** 2 + ((yy - center[1]) / 54) ** 2))
    total = np.ones((224, 224), dtype=np.float32) + 3 * central
    accumulated = base * total[..., None]
    target_triangles = xy[triangles]
    target_edges = np.roll(target_triangles, -1, axis=1) - target_triangles
    target_lengths = np.linalg.norm(target_edges, axis=2)
    for photo in candidates:
        if photo is anchor:
            continue
        source = photo.landmarks[:468, :2].astype(np.float32)
        source_triangles = source[triangles]
        source_edges = np.roll(source_triangles, -1, axis=1) - source_triangles
        ratio = (np.linalg.norm(source_edges, axis=2) + 0.5) / (target_lengths + 0.5)
        # Downweight foreshortened or heavily distorted regions.
        weights = np.exp(-2 * np.mean(np.abs(np.log(ratio)), axis=1))
        map_x, map_y, weight = triangle_maps(source, xy, triangles, 224, 224, weights)
        overlap = weight > 0.25
        if np.count_nonzero(overlap) < 100:
            continue
        warped = cv2.remap(photo.rgb, map_x, map_y, cv2.INTER_LINEAR).astype(np.float32)
        src, dst = warped[overlap], base[overlap]
        gain = np.clip(dst.std(axis=0) / np.maximum(src.std(axis=0), 8), 0.75, 1.3)
        corrected = np.clip((warped - src.mean(axis=0)) * gain + dst.mean(axis=0), 0, 255)
        sharpness = np.clip(photo.sharpness / max(anchor.sharpness, 1), 0.08, 1.5)
        feather = cv2.GaussianBlur((weight > 0).astype(np.float32), (0, 0), 3)
        weight *= feather * sharpness * (1 - 0.65 * central)
        accumulated += corrected * weight[..., None]
        total += weight
    texture = np.rint(accumulated / total[..., None]).clip(0, 255).astype(np.uint8)
    return texture, points, triangles


def project_mesh(points, yaw, pitch, size):
    """Perspective-project the observed facial surface around its central depth."""
    xyz = points - (points.min(axis=0) + points.max(axis=0)) / 2
    y, p = radians(yaw), radians(pitch)
    x_rot = xyz[:, 0] * cos(y) + xyz[:, 2] * sin(y)
    z_rot = -xyz[:, 0] * sin(y) + xyz[:, 2] * cos(y)
    y_rot = xyz[:, 1] * cos(p) - z_rot * sin(p)
    depth = xyz[:, 1] * sin(p) + z_rot * cos(p)
    factor = 0.78 * size / max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), 1)
    perspective = 400 / np.maximum(400 + depth, 80)
    projected = np.column_stack((x_rot, y_rot)) * (factor * perspective[:, None]) + size / 2
    return projected.astype(np.float32), depth


def render_mesh(texture, points, triangles, yaw, pitch, size, wireframe):
    projected, depth = project_mesh(points, yaw, pitch, size)
    order = np.argsort(-depth[triangles].mean(axis=1))
    faces = triangles[order]
    mapping_x, mapping_y, coverage = triangle_maps(points[:, :2], projected, faces, size, size)
    color = cv2.remap(texture, mapping_x, mapping_y, cv2.INTER_LINEAR)
    if wireframe:
        vertices = np.rint(projected).astype(np.int32)
        for face in faces:
            xy = projected[face]
            if cv2.contourArea(xy, oriented=True) * cv2.contourArea(points[face, :2], oriented=True) > 0:
                cv2.polylines(color, [vertices[face]], True, (85, 222, 193), 1, cv2.LINE_AA)
        coverage = np.maximum(coverage, np.any(color != 0, axis=2))
    return np.dstack((color, np.uint8(coverage > 0) * 255))
