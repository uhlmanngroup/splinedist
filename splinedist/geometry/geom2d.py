from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cv2
import numpy as np
import tensorflow as tf
from csbdeep.utils import _raise
from scipy.interpolate import interp1d
from skimage.draw import polygon
from skimage.measure import regionprops

from .. import splinegenerator as sg
from ..matching import _check_label_array
from ..utils import _normalize_grid


def spline_dist(a, n_params=32, contoursize_max=400):
    """'a' assumbed to be a label image with integer values that encode object ids. id 0 denotes background."""

    n_params >= 3 or _raise(ValueError("need 'n_params' >= 3"))

    dist = np.zeros((a.shape[0], a.shape[1], contoursize_max, 2))

    obj_list = np.unique(a)
    obj_list = obj_list[1:]

    for i in range(len(obj_list)):
        mask_temp = a.copy()
        mask_temp[mask_temp != obj_list[i]] = 0
        mask_temp[mask_temp > 0] = 1

        contour = contour_cv2_mask_uniform(mask_temp, contoursize_max)
        idx_nonzero = np.argwhere(mask_temp)
        dist[idx_nonzero[:, 0], idx_nonzero[:, 1]] = contour

    dist = np.reshape(dist, (dist.shape[0], dist.shape[1], -1))
    return dist


def dist_to_coord(rhos, grid=(1, 1)):
    """convert from polar to cartesian coordinates for a single image (3-D array) or multiple images (4-D array)"""
    grid = _normalize_grid(grid, 2)
    is_single_image = rhos.ndim == 3
    if is_single_image:
        rhos = np.expand_dims(rhos, 0)
    assert rhos.ndim == 4

    rhos = np.reshape(rhos, (rhos.shape[0], rhos.shape[1], rhos.shape[2], -1, 2))

    n_images, h, w, n_params, _ = rhos.shape
    coord = np.empty((n_images, h, w, 2, n_params), dtype=rhos.dtype)

    start = np.indices((h, w))
    for i in range(2):
        coord[..., i, :] = grid[i] * np.broadcast_to(
            start[i].reshape(1, h, w, 1), (n_images, h, w, n_params)
        )

    # phis = 2 * math.pi * expit(rhos[:,:,:,:,1])
    phis = rhos[:, :, :, :, 1]
    rhos = rhos[:, :, :, :, 0]

    coord[..., 0, :] += rhos * np.cos(phis)  # row coordinate
    coord[..., 1, :] += rhos * np.sin(phis)  # col coordinate
    return coord[0] if is_single_image else coord


def polygons_to_label(coord, prob, points, shape=None, thr=-np.inf):
    sh = coord.shape[:2] if shape is None else shape
    lbl = np.zeros(sh, np.int32)
    # sort points with increasing probability

    ind = np.argsort([prob[p[0], p[1]] for p in points])
    points = points[ind]

    M = coord.shape[3]
    phi = np.load("./phi_" + str(M) + ".npy")
    phi = tf.convert_to_tensor(phi)

    i = 1
    for p in points:
        if prob[p[0], p[1]] < thr:
            continue
        coefs = coord[p[0], p[1]]
        coefs = np.transpose(coefs, (1, 0))
        contour = sg.SplineCurveVectorized(M, sg.B3(), True, coefs)
        contour = contour.sampleSequential(phi)
        contour = contour.numpy()

        rr, cc = polygon(contour[:, 0], contour[:, 1], sh)
        lbl[rr, cc] = i

        X = np.clip(contour[:, 0], 0, lbl.shape[0] - 1)
        Y = np.clip(contour[:, 1], 0, lbl.shape[1] - 1)
        lbl[np.rint(X).astype(int), np.rint(Y).astype(int)] = i

        i += 1
    return lbl


def ray_angles(n_params=32):
    return np.linspace(0, 2 * np.pi, n_params, endpoint=False)


def relabel_image_splinedist(lbl, n_params, **kwargs):
    """relabel each label region in `lbl` with its spline representation"""
    _check_label_array(lbl, "lbl")
    if not lbl.ndim == 2:
        raise ValueError("lbl image should be 2 dimensional")
    dist = spline_dist(lbl, n_params, **kwargs)
    coord = dist_to_coord(dist)
    points = np.array(tuple(np.array(r.centroid).astype(int) for r in regionprops(lbl)))
    return polygons_to_label(coord, np.ones_like(lbl), points, shape=lbl.shape)


def contour_cv2_mask_uniform(mask, contoursize_max):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    max_ind = np.argmax(areas)
    contour = np.squeeze(contours[max_ind])
    contour = np.reshape(contour, (-1, 2))
    contour = np.append(contour, contour[0].reshape((-1, 2)), axis=0)
    contour = contour.astype("float32")

    rows, cols = mask.shape
    delta = np.diff(contour, axis=0)
    s = [0]
    for d in delta:
        dl = s[-1] + np.linalg.norm(d)
        s.append(dl)

    if s[-1] == 0:
        s[-1] = 1

    s = np.array(s) / s[-1]
    fx = interp1d(s, contour[:, 0] / rows, kind="linear")
    fy = interp1d(s, contour[:, 1] / cols, kind="linear")
    S = np.linspace(0, 1, contoursize_max, endpoint=False)
    X = rows * fx(S)
    Y = cols * fy(S)

    contour = np.transpose(np.stack([X, Y])).astype(np.float32)

    contour = np.stack((contour[:, 1], contour[:, 0]), axis=-1)
    return contour
