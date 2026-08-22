"""Ref2VA multi-reference enablement, verified against the built image.

The image must contain the patches/minimax_h3_pipeline.py overwrite and the
allow-mixed-ref-inputs API patch; these tests fail closed on a stale image.
"""

import importlib
import inspect
import typing

import pytest
from PIL import Image

pipeline = importlib.import_module("vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3")


def test_pipeline_overwrite_is_installed():
    assert callable(pipeline._load_images)
    assert callable(pipeline._load_audios)
    assert callable(pipeline._cat_rows)


def test_load_images_coerces_none_single_and_many():
    img = Image.new("RGB", (64, 64))
    assert pipeline._load_images(None) == []
    assert len(pipeline._load_images(img)) == 1
    assert len(pipeline._load_images([img, img, img])) == 3


def test_load_audios_coerces_single_and_many():
    pytest.importorskip("torchaudio")
    import torch

    waveform = torch.zeros(1, 3200)
    assert pipeline._load_audios(None) == []
    # A bare (waveform, sample_rate) tuple IS a sequence; pass items in a list.
    single = pipeline._load_audios([(waveform, 32000)])
    assert len(single) == 1 and single[0][1] == 32000
    many = pipeline._load_audios([(waveform, 32000), {"waveform": waveform, "sample_rate": 16000}])
    assert [item[1] for item in many] == [32000, 16000]
    with pytest.raises(TypeError):
        pipeline._load_audios([object()])


def test_api_patch_allows_mixed_reference_containers():
    serving_video = importlib.import_module("vllm_omni.entrypoints.openai.serving_video")
    image_hints = typing.get_type_hints(serving_video.ReferenceImage)
    assert list[Image.Image] in typing.get_args(image_hints["data"])
    audio_hints = typing.get_type_hints(serving_video.ReferenceAudio)
    assert list[typing.Any] in typing.get_args(audio_hints["path"])

    api_server = importlib.import_module("vllm_omni.entrypoints.openai.api_server")
    source = inspect.getsource(api_server._parse_video_form)
    assert "is_image_upload" in source
    assert "validate_ref_counts" in source
