from dataclasses import replace

import pytest
from PIL import Image

from localimgsearch.media import (
    MediaKind,
    SamplingPolicy,
    extract_visual_units,
    probe_media,
)


def test_extensionless_still_is_probed_and_extracted_as_rgb(tmp_path):
    path = tmp_path / "extensionless"
    Image.new("RGBA", (7, 5), (255, 0, 0, 128)).save(path, format="PNG")

    info = probe_media(path)
    units = list(extract_visual_units(path, media_info=info))

    assert info.kind is MediaKind.IMAGE
    assert info.format_name == "PNG"
    assert (info.width, info.height) == (7, 5)
    assert len(units) == 1
    assert units[0].source_path == path.resolve()
    assert units[0].image.mode == "RGB"
    assert units[0].image.size == (7, 5)
    assert units[0].image.getpixel((0, 0)) == (255, 127, 127)


def test_gif_sampling_is_deterministic_capped_and_retains_endpoints(tmp_path):
    path = tmp_path / "animation.gif"
    colors = ["red", "green", "blue", "yellow", "purple", "cyan"]
    frames = [Image.new("RGB", (4, 3), color) for color in colors]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=[50, 950, 50, 950, 50, 950],
        disposal=[1] * len(frames),
        loop=0,
    )
    policy = SamplingPolicy(maximum_gap_seconds=0.4, maximum_units=4)
    info = probe_media(path)

    first = list(extract_visual_units(path, media_info=info, policy=policy))
    second = list(extract_visual_units(path, media_info=info, policy=policy))

    assert info.kind is MediaKind.GIF
    assert info.duration_seconds == pytest.approx(3.0)
    assert len(first) == policy.maximum_units
    assert first[0].source_frame_index == 0
    assert first[-1].source_frame_index == len(frames) - 1
    assert [unit.ordinal for unit in first] == list(range(len(first)))
    assert [unit.source_frame_index for unit in first] == [
        unit.source_frame_index for unit in second
    ]
    assert [unit.timestamp_seconds for unit in first] == [
        unit.timestamp_seconds for unit in second
    ]
    assert all(unit.image.mode == "RGB" for unit in first)
    assert all(unit.policy_fingerprint == policy.fingerprint for unit in first)

    # Frames own their pixels after Pillow advances and closes the GIF decoder.
    assert first[0].image.getpixel((0, 0)) == (255, 0, 0)
    assert first[-1].image.getpixel((0, 0)) == (0, 255, 255)


@pytest.mark.parametrize(
    "changes",
    [
        {"maximum_gap_seconds": 0},
        {"maximum_gap_seconds": float("inf")},
        {"maximum_units": 0},
        {"video_end_margin_seconds": -0.1},
        {"alpha_background": (0, 0, 256)},
    ],
)
def test_policy_rejects_invalid_settings(changes):
    with pytest.raises(ValueError):
        SamplingPolicy(**changes)


def test_policy_fingerprint_is_stable_and_sensitive_to_settings():
    policy = SamplingPolicy()

    assert policy.fingerprint == SamplingPolicy().fingerprint
    assert policy.fingerprint != replace(policy, maximum_units=8).fingerprint
    assert len(policy.fingerprint) == 64
