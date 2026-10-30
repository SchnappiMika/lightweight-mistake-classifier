import hashlib
import json
import socket

import pytest

from app.core.preprocess.formula_recognition_backend import (REQUIRED, normalize_options,
    offline_network, prepare_image, recognize, validate_models)


def model_manifest(directory):
    entries = []
    for name in REQUIRED:
        content = ("fixture " + name).encode()
        (directory / name).write_bytes(content)
        entries.append({"filename": name, "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()})
    (directory / "manifest.json").write_text(json.dumps({"files": entries}), encoding="utf-8")


def test_formula_model_validation_checks_every_file(tmp_path):
    model_manifest(tmp_path)
    validate_models(tmp_path)
    original = (tmp_path / "weights.pth").read_bytes()
    (tmp_path / "weights.pth").write_bytes(b"X" * len(original))
    with pytest.raises(ValueError, match="校验失败"):
        validate_models(tmp_path)


def test_formula_missing_model_fails_before_network_or_inference(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(b"image fixture")
    result = recognize(source, tmp_path)
    assert result["error"]
    assert not result["text"]
    assert not result["confirmed"]
    assert result["media_hash"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_offline_formula_guard_blocks_and_restores_socket():
    original = socket.create_connection
    original_connect = socket.socket.connect
    with offline_network():
        with pytest.raises(RuntimeError, match="禁止联网"):
            socket.create_connection(("example.com", 443))
        with socket.socket() as connection:
            with pytest.raises(RuntimeError, match="禁止联网"):
                connection.connect(("127.0.0.1", 80))
    assert socket.create_connection is original
    assert socket.socket.connect is original_connect


def test_crop_invert_and_fit_keep_original_bytes(tmp_path):
    from PIL import Image
    source = tmp_path / "original.png"
    Image.new("RGB", (800, 400), "black").save(source)
    original = source.read_bytes()
    image, metadata = prepare_image(source, {"crop_box": [20, 30, 530, 240],
                                             "invert": True, "resize_mode": "fit"})
    assert image.size == (466, 192)
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert metadata["original_size"] == [800, 400]
    assert metadata["selected_size"] == [510, 210]
    assert source.read_bytes() == original


def test_original_mode_preserves_crop_size_and_orientation(tmp_path):
    from PIL import Image
    source = tmp_path / "source.png"
    original = Image.new("RGB", (10, 8), "white")
    original.putpixel((2, 3), (1, 2, 3))
    original.save(source)
    image, metadata = prepare_image(source, {"crop_box": [2, 3, 8, 7], "resize_mode": "original"})
    assert image.size == (6, 4)
    assert image.getpixel((0, 0)) == (1, 2, 3)
    assert metadata["options"]["invert"] is False


@pytest.mark.parametrize("options", [
    {"crop_box": [-1, 0, 5, 5]}, {"crop_box": [1, 1, 1, 5]},
    {"crop_box": [0, 0, 99, 99]}, {"crop_box": [0, 0, 1.5, 5]},
    {"crop_box": [0, 0, 1, 1]}, {"crop_box": [0, 0, float('nan'), 5]},
    {"invert": "false"}, {"resize_mode": "unbounded"}, [1, 2],
])
def test_invalid_options_rejected_before_model_load(tmp_path, options):
    from PIL import Image
    source = tmp_path / "sample.png"
    Image.new("RGB", (10, 10), "white").save(source)
    record = recognize(source, tmp_path, options)
    assert record["error"]
    assert record["media_hash"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record["text"] == ""


def test_bad_image_returns_original_hash_and_error(tmp_path):
    source = tmp_path / "corrupt.png"
    source.write_bytes(b"broken image")
    record = recognize(source, tmp_path, {"crop_box": [0, 0, 5, 5]})
    assert "图片损坏" in record["error"]
    assert record["media_hash"] == hashlib.sha256(source.read_bytes()).hexdigest()
