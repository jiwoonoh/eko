"""Host-resident R-U-Net ECG delineator.

R-U-Net: Scharpf et al., "Decoder Design Matters for ECG Delineation" (arXiv:2609.16489).
Ports ELM-Research/semi-seg-ecg's resnet.py, unet_head.py, and transforms.py.
Parameter names match its checkpoints.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample, sosfiltfilt
import torch
from torch import nn

from ...leads import DEFAULT_MEASUREMENT_LEAD, LEADS
from ...record import load_record

# semi-seg-ecg's label encoding. 0 is background.
WAVES = {1: "P", 2: "QRS", 3: "T"}

# semi-seg-ecg's HighpassFilter and LowpassFilter default order.
FILTER_ORDER = 5


class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, 3, stride=stride, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18(nn.Module):
    """Single-lead encoder returning every stage's feature map."""

    def __init__(self, num_leads: int, strides: list[int], dilations: list[int]) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(num_leads, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)

        inplanes = 64
        self.stages = []
        for index, (stride, dilation) in enumerate(zip(strides, dilations)):
            planes = 64 * 2 ** index
            stage = nn.Sequential(BasicBlock(inplanes, planes, stride, dilation),
                                  BasicBlock(planes, planes, 1, dilation))
            self.add_module(f"layer{index + 1}", stage)
            self.stages.append(stage)
            inplanes = planes

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.maxpool(self.stem(x))
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetHead(nn.Module):
    """Upsample from the deepest feature map, fusing one encoder skip per stage."""

    def __init__(self, in_channels: list[int], channels: list[int], num_classes: int,
                 kernel_size: int, align_corners: bool) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.decoder_blocks = nn.ModuleList()
        previous = in_channels[-1]
        for skip, out_channels in zip(reversed(in_channels[:-1]), channels):
            self.decoder_blocks.append(DoubleConv(previous + skip, out_channels, kernel_size))
            previous = out_channels
        # Inference omits semi-seg-ecg's training-only dropout before this layer.
        self.cls_seg = nn.Conv1d(previous, num_classes, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        for block, skip in zip(self.decoder_blocks, reversed(features[:-1])):
            x = nn.functional.interpolate(x, size=skip.shape[-1], mode="linear",
                                          align_corners=self.align_corners)
            x = block(torch.cat([x, skip], dim=1))
        return self.cls_seg(x)


class RUNet(nn.Module):
    def __init__(self, backbone: dict, decode_head: dict) -> None:
        super().__init__()
        self.backbone = ResNet18(backbone["num_leads"], backbone["strides"],
                                 backbone["dilations"])
        self.decode_head = UNetHead(decode_head["in_channels"], decode_head["channels"],
                                    decode_head["num_classes"], decode_head["kernel_size"],
                                    decode_head["align_corners"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample class logits at the input length."""
        logits = self.decode_head(self.backbone(x))
        return nn.functional.interpolate(logits, size=x.shape[-1], mode="linear",
                                         align_corners=self.decode_head.align_corners)


def check_config(cfg: dict) -> tuple[dict, dict]:
    """Return backbone and head settings, rejecting architectures this file lacks."""
    backbone = cfg["backbone"].get("resnet18")
    decode_head = cfg["decode_head"].get("UNetHead")
    if backbone is None or decode_head is None:
        raise ValueError("R-U-Net checkpoints must use a resnet18 backbone and UNetHead")
    if (backbone["num_leads"] != 1 or backbone["deep_stem"] or backbone["avg_down"]
            or backbone["contract_dilation"]):
        raise ValueError("R-U-Net delineator supports the single-lead default ResNet18 stem")
    if backbone["out_indices"] != [0, 1, 2, 3] or decode_head["num_classes"] != len(WAVES) + 1:
        raise ValueError("R-U-Net delineator expects four encoder stages and P/QRS/T classes")
    return backbone, decode_head


def model_rate(cfg: dict) -> int:
    """Return the training sampling rate, which the dataset filters record."""
    rates = {next(iter(step.values()))["fs"] for step in cfg["dataset"]["filter"]}
    if len(rates) != 1:
        raise ValueError(f"Checkpoint filters disagree on sampling rate: {sorted(rates)}")
    return rates.pop()


class Delineator:
    def __init__(self, checkpoint: Path, device: str | None = None) -> None:
        state = torch.load(checkpoint.expanduser(), map_location="cpu", weights_only=False)
        cfg = state["config"]
        backbone, decode_head = check_config(cfg)

        self.sampling_rate = model_rate(cfg)
        self.filters = [
            butter(FILTER_ORDER, settings["cutoff"], btype=name.removesuffix("_filter"),
                   fs=settings["fs"], output="sos")
            for step in cfg["dataset"]["filter"] for name, settings in step.items()
        ]

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = RUNet(backbone, decode_head)
        # semi-seg-ecg's inference.py evaluates the student, not the EMA teacher.
        self.model.load_state_dict(state["model"], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)

    def preprocess(self, signal: np.ndarray, rate: float) -> np.ndarray:
        """Resample, filter, then standardize, in semi-seg-ecg's order."""
        signal = resample(signal, round(signal.size * self.sampling_rate / rate))
        for sos in self.filters:
            signal = sosfiltfilt(sos, signal)
        scale = signal.std()
        if scale == 0:
            raise ValueError("Lead is constant after filtering")
        return (signal - signal.mean()) / scale

    @torch.inference_mode()
    def labels(self, signal: np.ndarray, rate: float) -> np.ndarray:
        """Return a wave class for every sample at the record's rate."""
        source = torch.as_tensor(self.preprocess(signal, rate), dtype=torch.float32)
        logits = self.model(source[None, None].to(self.device))
        # Resize logits, not labels, so classes are chosen at the record's samples.
        logits = nn.functional.interpolate(logits, size=signal.size, mode="linear",
                                           align_corners=self.model.decode_head.align_corners)
        return logits.argmax(dim=1)[0].cpu().numpy()

    def evaluate(self, ecg_path: str | Path, lead: str = DEFAULT_MEASUREMENT_LEAD) -> dict:
        """Return P, QRS, and T onset/offset sample indices for one lead."""
        if lead not in LEADS:
            raise ValueError(f"lead must be one of {', '.join(LEADS)}")
        data, rate = load_record(ecg_path)
        signal = np.asarray(data["feats"], dtype=float)
        if signal.ndim != 2 or signal.shape[0] != len(LEADS):
            raise ValueError(f"Expected (12, samples) in lead order {LEADS}")

        signal = signal[LEADS.index(lead)]
        if not np.isfinite(signal).all():
            raise ValueError(f"Lead {lead} contains nonfinite values")
        return {"lead": lead, "sampling_rate": rate,
                "waves": wave_spans(self.labels(signal, rate))}


def wave_spans(labels: np.ndarray) -> dict[str, list[list[int]]]:
    """Group runs of one class into inclusive [onset, offset] sample indices.

    Drops runs cut off by either end of the recording.
    """
    edges = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [labels.size])) - 1

    spans = {wave: [] for wave in WAVES.values()}
    for start, end in zip(starts, ends):
        wave = WAVES.get(int(labels[start]))
        if wave and start > 0 and end < labels.size - 1:
            spans[wave].append([int(start), int(end)])
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", help="PyTorch device (default: CUDA when available, else CPU)")
    parser.add_argument("--lead", choices=LEADS, default=DEFAULT_MEASUREMENT_LEAD)
    parser.add_argument("ecg_path", type=Path)
    args = parser.parse_args()
    delineator = Delineator(args.checkpoint, args.device)
    print(json.dumps(delineator.evaluate(args.ecg_path, args.lead), allow_nan=False))


if __name__ == "__main__":
    main()
