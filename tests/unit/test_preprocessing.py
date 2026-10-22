import io

import numpy as np
import pytest
from PIL import Image

from app.core.errors import ImageValidationException
from app.services.image_service import detect_image_mime, preprocess_image


def test_detect_image_mime():
    # Valid JPEG magic bytes
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    assert detect_image_mime(jpeg_bytes) == "image/jpeg"

    # Valid PNG magic bytes
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00"
    assert detect_image_mime(png_bytes) == "image/png"

    # Non-image bytes
    assert detect_image_mime(b"<!DOCTYPE html>") is None
    assert detect_image_mime(b"") is None


def test_preprocess_image_valid_rgb():
    img = Image.new("RGB", (300, 300), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    tensor = preprocess_image(img_bytes, target_size=(224, 224), interpolation="nearest")
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() <= tensor.max() <= 1.0


def test_preprocess_image_rgba_and_grayscale():
    # RGBA image conversion to 3 channels
    rgba_img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    tensor_rgba = preprocess_image(rgba_img, target_size=(224, 224))
    assert tensor_rgba.shape == (1, 224, 224, 3)

    # Grayscale image conversion to 3 channels
    gray_img = Image.new("L", (100, 100), color=128)
    tensor_gray = preprocess_image(gray_img, target_size=(224, 224))
    assert tensor_gray.shape == (1, 224, 224, 3)


def test_preprocess_image_modes():
    img = Image.new("RGB", (100, 100), color=(50, 100, 150))
    for mode in ["nearest", "bicubic", "bilinear"]:
        tensor = preprocess_image(img, interpolation=mode)
        assert tensor.shape == (1, 224, 224, 3)


def test_preprocess_image_empty_and_corrupt():
    with pytest.raises(ImageValidationException):
        preprocess_image(b"")

    with pytest.raises(ImageValidationException):
        preprocess_image(b"not an image data bytes")


def test_preprocess_image_decompression_bomb_guard():
    # Attempting to process image with pixel count over safety limit
    huge_img = Image.new("RGB", (2000, 2000))
    with pytest.raises(ImageValidationException):
        preprocess_image(huge_img, max_pixels=1000)
