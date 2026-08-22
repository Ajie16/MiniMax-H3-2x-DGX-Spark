from h3_multinode.ref_inputs import (
    MAX_REF_AUDIOS,
    MAX_REF_IMAGES,
    MAX_REF_VIDEOS,
    coerce_ref_sequence,
    is_audio_upload,
    is_image_upload,
    is_video_upload,
    ref2va_condition_labels,
    validate_ref_counts,
)


def test_coerce_ref_sequence_unwraps_none_one_and_many():
    assert coerce_ref_sequence(None) == []
    assert coerce_ref_sequence("a") == ["a"]
    assert coerce_ref_sequence(["a", None, "b"]) == ["a", "b"]


def test_upload_sniff_by_mime_and_name():
    assert is_image_upload("a.PNG", None)
    assert is_image_upload("x", "image/jpeg")
    assert is_audio_upload("a.wav", None)
    assert is_audio_upload("x", "audio/mpeg")
    assert is_video_upload("clip.mp4", None)
    assert is_video_upload("x", "video/webm")
    assert not is_image_upload("clip.mp4", "video/mp4")


def test_ref2va_labels_match_comfy_order():
    assert ref2va_condition_labels(n_images=2, video_has_audio=(True, False), n_audios=1) == [
        ("image", 1),
        ("image", 2),
        ("audio", 1),
        ("video", 1),
        ("video", 2),
        ("audio", 2),
    ]
    assert ref2va_condition_labels(n_images=1, n_audios=0) == [("image", 1)]


def test_validate_ref_counts_accepts_comfy_caps():
    validate_ref_counts(MAX_REF_IMAGES, MAX_REF_VIDEOS, MAX_REF_AUDIOS)
    validate_ref_counts(0, 0, 0)
    validate_ref_counts(2, 1, 1)


def test_validate_ref_counts_rejects_over_cap():
    import pytest

    with pytest.raises(ValueError, match="reference images"):
        validate_ref_counts(MAX_REF_IMAGES + 1, 0, 0)
    with pytest.raises(ValueError, match="reference videos"):
        validate_ref_counts(0, MAX_REF_VIDEOS + 1, 0)
    with pytest.raises(ValueError, match="standalone audios"):
        validate_ref_counts(0, 0, MAX_REF_AUDIOS + 1)
