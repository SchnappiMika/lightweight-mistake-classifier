"""Protect formula detail when fitting the official checkpoint's input bounds."""
from pathlib import Path

import pytest
from PIL import Image

from app.core.preprocess.formula_recognition_backend import prepare_image, sha256


SAMPLES = Path(__file__).resolve().parents[1] / "samples" / "public_acceptance" / "media"


@pytest.mark.parametrize("height", [150, 151, 192])
def test_fit_preserves_supported_size_pixels_including_small_marks(tmp_path, height):
    source = tmp_path / "fine_symbols.png"
    original = Image.new("RGB", (354, height), "white")
    # Tiny exponent-like marks must survive unchanged when no model limit is exceeded.
    original.putpixel((120, 6), (0, 0, 0))
    original.putpixel((121, 7), (0, 0, 0))
    original.save(source)
    before = source.read_bytes()

    prepared, metadata = prepare_image(source)

    assert prepared.size == original.size
    assert prepared.tobytes() == original.tobytes()
    assert metadata["processed_size"] == [354, height]
    assert source.read_bytes() == before


@pytest.mark.parametrize("name,options,expected_size", [
    ("formula_leibniz_notation.png", {"resize_mode": "fit"}, (336, 192)),
    ("formula_center_of_mass.png",
     {"crop_box": [40, 40, 550, 250], "invert": True, "resize_mode": "fit"}, (466, 192)),
])
def test_public_fraction_and_integral_use_checkpoint_bounds_without_changing_source(name, options, expected_size):
    source = SAMPLES / name
    before = sha256(source)
    prepared, metadata = prepare_image(source, options)

    assert prepared.size == expected_size
    assert metadata["processed_size"] == list(expected_size)
    assert sha256(source) == before


@pytest.mark.parametrize("size,expected", [
    ((1344, 192), (672, 96)),
    ((672, 384), (336, 192)),
    ((1344, 384), (672, 192)),
    ((32, 16), (32, 16)),
])
def test_fit_respects_both_input_limits_without_upscaling(tmp_path, size, expected):
    source = tmp_path / "formula.png"
    Image.new("RGB", size, "white").save(source)
    prepared, _ = prepare_image(source, {"resize_mode": "fit"})
    assert prepared.size == expected


def test_original_mode_still_bypasses_external_size_limits(tmp_path):
    source = tmp_path / "large.png"
    Image.new("RGB", (800, 400), "white").save(source)
    prepared, _ = prepare_image(source, {"resize_mode": "original"})
    assert prepared.size == (800, 400)
