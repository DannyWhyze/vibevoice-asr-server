from pathlib import Path
import sys
from unittest.mock import MagicMock
import pytest
import torch

# Ensure repository root is on sys.path for test runners
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legacy_scripts.test_local import decode_chunk_with_fallback, parse_args


def apply_padding_mask_fix(inputs: dict) -> dict:
    """Pad attention mask by 3200 samples if its length is a multiple of 3200.

    Workaround for Transformers off-by-one audio feature downsampling bug.
    """
    if "padding_mask" in inputs:
        mask = inputs["padding_mask"]
        if mask.shape[-1] % 3200 == 0:
            extra_padding = torch.ones((mask.shape[0], 3200), dtype=mask.dtype, device=mask.device)
            inputs["padding_mask"] = torch.cat([mask, extra_padding], dim=-1)
    return inputs


def test_padding_mask_fix_applied_when_multiple_of_3200():
    mask = torch.ones((1, 4500 * 3200), dtype=torch.bfloat16)
    inputs = {"padding_mask": mask}
    processed = apply_padding_mask_fix(inputs)
    assert processed["padding_mask"].shape[-1] == 4501 * 3200


def test_padding_mask_fix_skipped_when_not_multiple_of_3200():
    mask = torch.ones((1, 3201), dtype=torch.bfloat16)
    inputs = {"padding_mask": mask}
    processed = apply_padding_mask_fix(inputs)
    assert processed["padding_mask"].shape[-1] == 3201


def test_timestamp_offset_math():
    raw_segments = [{"Start": 1.5, "End": 5.0, "Speaker": "1", "Content": "Hello"}]
    time_offset = 600.0  # 10 minutes
    adjusted = []
    for s in raw_segments:
        item = s.copy()
        item["Start"] = item.get("Start", 0.0) + time_offset
        item["End"] = item.get("End", 0.0) + time_offset
        adjusted.append(item)
    assert adjusted[0]["Start"] == 601.5
    assert adjusted[0]["End"] == 605.0


def test_cli_argument_parsing():
    args = parse_args(["--audio", "sample.wav"])
    assert args.audio == "sample.wav"
    assert args.minutes == 10
    assert args.output is None

    custom_args = parse_args(["-a", "sample.wav", "-m", "15", "-o", "transcript.json"])
    assert custom_args.audio == "sample.wav"
    assert custom_args.minutes == 15
    assert custom_args.output == "transcript.json"


def test_decode_chunk_with_fallback_success():
    mock_processor = MagicMock()
    mock_processor.decode.return_value = [[
        {"Start": 1.0, "End": 5.0, "Speaker": "1", "Content": "Hello world"}
    ]]
    dummy_ids = torch.tensor([1, 2, 3])

    result = decode_chunk_with_fallback(mock_processor, dummy_ids, block_duration=5.0)

    assert mock_processor.decode.call_count == 1
    call_args, call_kwargs = mock_processor.decode.call_args
    assert torch.equal(call_args[0], dummy_ids)
    assert call_kwargs.get("return_format") == "parsed"
    assert len(result) == 1
    assert result[0]["Content"] == "Hello world"
    assert result[0]["Start"] == 1.0
    assert result[0]["End"] == 5.0


def test_decode_chunk_with_fallback_on_json_error():
    mock_processor = MagicMock()

    def mock_decode(token_ids, return_format="parsed"):
        if return_format == "parsed":
            raise ValueError("Unterminated string starting at line 1 column 500")
        return ["Fallback raw text content"]

    mock_processor.decode.side_effect = mock_decode

    result = decode_chunk_with_fallback(
        mock_processor, torch.tensor([1, 2, 3]), block_duration=600.0, chunk_index=2
    )

    assert len(result) == 1
    assert result[0]["Start"] == 0.0
    assert result[0]["End"] == 600.0
    assert result[0]["Speaker"] == "Unknown"
    assert result[0]["Content"] == "Fallback raw text content"
