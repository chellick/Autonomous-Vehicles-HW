import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def brightness(image, alpha):
    if alpha == 0:
        return image
    return ImageEnhance.Brightness(image).enhance(2 ** alpha)


def contrast(image, alpha):
    if alpha == 1.0:
        return image
    return ImageEnhance.Contrast(image).enhance(alpha)


def gamma(image, alpha):
    if alpha == 1.0:
        return image
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = np.clip(arr ** alpha, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def gaussian_noise(image, alpha):
    if alpha == 0:
        return image
    rng = np.random.default_rng(42)
    arr = np.asarray(image).astype(np.float32)
    noise = rng.normal(0, alpha, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def gaussian_blur(image, alpha):
    if alpha == 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=alpha))


def jpeg_quality(image, alpha):
    if alpha >= 100:
        return image
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(alpha))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def occlusion(image, alpha):
    if alpha == 0:
        return image
    w, h = image.size
    side = int(((alpha / 100) * w * h) ** 0.5)
    rng = random.Random(42)
    x = rng.randint(0, max(0, w - side))
    y = rng.randint(0, max(0, h - side))
    arr = np.asarray(image).copy()
    arr[y:y + side, x:x + side] = 128
    return Image.fromarray(arr)


TRANSFORMS = {
    "brightness": (brightness, [-0.5, -0.25, 0, 0.25, 0.5], "EV"),
    "contrast": (contrast, [0.5, 0.75, 1.0, 1.25, 1.5], "factor"),
    "gamma": (gamma, [0.5, 0.75, 1.0, 1.5, 2.0], "exponent"),
    "noise": (gaussian_noise, [0, 5, 10, 20, 40], "sigma_uint8"),
    "blur": (gaussian_blur, [0, 1, 2, 4, 8], "radius_px"),
    "jpeg": (jpeg_quality, [10, 20, 40, 60, 80, 100], "quality"),
    "occlusion": (occlusion, [0, 5, 10, 20, 30], "percent_area"),
}


INVARIANT_RANGES = {
    "brightness": [-0.25, 0.25],
    "contrast": [0.75, 1.25],
    "gamma": [0.75, 1.5],
    "jpeg": [80, 100],
}
