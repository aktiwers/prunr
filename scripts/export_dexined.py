#!/usr/bin/env python3
"""Export DexiNed from PyTorch to ONNX (compatible with ORT 2.0).

Downloads weights from HuggingFace, exports clean FP32 ONNX with no quantization.
"""
import sys
import urllib.request
from pathlib import Path

# DexiNed model definition (inline from the official repo to avoid cloning)
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoFusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.GroupNorm(4, 64)
        self.norm_layer2 = nn.GroupNorm(4, 64)

    def forward(self, x):
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = self.relu(self.norm_layer2(self.conv2(attn)))
        attn = F.softmax(self.conv3(attn), dim=1)
        return ((x * attn).sum(1)).unsqueeze(1)


class _DenseLayer(nn.Module):
    def __init__(self, input_features, out_features):
        super().__init__()
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_features, out_features, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat([x, self.dense_layer(x)], 1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, input_features, out_features):
        super().__init__()
        self.layers = nn.ModuleList([
            _DenseLayer(input_features + i * out_features, out_features)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, use_bs=True):
        super().__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features, out_features=None, stride=1, use_act=True):
        super().__init__()
        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_features)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        if self.use_act:
            x = self.relu(x)
        return x


class DexiNed(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = DoubleConvBlock(3, 32, 64, stride=2)
        self.block_2 = DoubleConvBlock(64, 128, stride=2)
        self.block_3 = DoubleConvBlock(128, 256, stride=2)
        self.block_4 = DoubleConvBlock(256, 512, stride=2)
        self.block_5 = DoubleConvBlock(512, 512, stride=2)
        self.block_6 = DoubleConvBlock(512, 256, stride=2)

        self.dblock_1 = _DenseBlock(2, 64, 32)
        self.dblock_2 = _DenseBlock(3, 128, 32)
        self.dblock_3 = _DenseBlock(3, 256, 32)
        self.dblock_4 = _DenseBlock(3, 512, 32)
        self.dblock_5 = _DenseBlock(3, 512, 32)
        self.dblock_6 = _DenseBlock(3, 256, 32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.side_1 = SingleConvBlock(128, 1, 1, use_bs=False)
        self.side_2 = SingleConvBlock(224, 1, 1, use_bs=False)
        self.side_3 = SingleConvBlock(352, 1, 1, use_bs=False)
        self.side_4 = SingleConvBlock(608, 1, 1, use_bs=False)
        self.side_5 = SingleConvBlock(608, 1, 1, use_bs=False)
        self.side_6 = SingleConvBlock(352, 1, 1, use_bs=False)

        self.pre_dense_3 = SingleConvBlock(256, 128, 2, use_bs=False)
        self.pre_dense_4 = SingleConvBlock(512, 256, 2, use_bs=False)
        self.pre_dense_5_0 = SingleConvBlock(512, 256, 2, use_bs=False)
        self.pre_dense_5 = SingleConvBlock(256, 128, 2, use_bs=False)
        self.pre_dense_6_0 = SingleConvBlock(256, 256, 2, use_bs=False)
        self.pre_dense_6 = SingleConvBlock(256, 128, 2, use_bs=False)

        self.up_block_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_block_2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_block_3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_block_4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.up_block_5 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.up_block_6 = nn.Upsample(scale_factor=64, mode='bilinear', align_corners=True)

        self.block_cat = CoFusion(6, 6)

    def slice_cat(self, x, block):
        h, w = x.shape[2], x.shape[3]
        result = block(x)
        return result[:, :, :h, :w]

    def forward(self, x):
        # Block 1
        b1 = self.block_1(x)
        db1 = self.dblock_1(b1)
        s1 = self.side_1(db1)

        # Block 2
        b2 = self.block_2(b1)
        db2 = self.dblock_2(b2)
        s2 = self.side_2(db2)

        # Block 3
        b3 = self.block_3(b2)
        db3_0 = self.pre_dense_3(b3)
        db3 = self.dblock_3(torch.cat([b3, db3_0], 1))
        s3 = self.side_3(db3)

        # Block 4
        b4 = self.block_4(b3)
        db4_0 = self.pre_dense_4(b4)
        db4 = self.dblock_4(torch.cat([b4, db4_0], 1))
        s4 = self.side_4(db4)

        # Block 5
        b5 = self.block_5(b4)
        db5_0 = self.pre_dense_5_0(b5)
        db5_1 = self.pre_dense_5(db5_0)
        db5 = self.dblock_5(torch.cat([b5, db5_1], 1))
        s5 = self.side_5(db5)

        # Block 6
        b6 = self.block_6(b5)
        db6_0 = self.pre_dense_6_0(b6)
        db6_1 = self.pre_dense_6(db6_0)
        db6 = self.dblock_6(torch.cat([b6, db6_1], 1))
        s6 = self.side_6(db6)

        # Upsample
        out_1 = self.up_block_1(s1)
        out_2 = self.up_block_2(s2)
        out_3 = self.up_block_3(s3)
        out_4 = self.up_block_4(s4)
        out_5 = self.up_block_5(s5)
        out_6 = self.up_block_6(s6)

        # Crop to input size
        h, w = x.shape[2], x.shape[3]
        out_1 = out_1[:, :, :h, :w]
        out_2 = out_2[:, :, :h, :w]
        out_3 = out_3[:, :, :h, :w]
        out_4 = out_4[:, :, :h, :w]
        out_5 = out_5[:, :, :h, :w]
        out_6 = out_6[:, :, :h, :w]

        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # Fuse
        block_cat = torch.cat(results, dim=1)
        block_cat = self.block_cat(block_cat)

        results.append(block_cat)
        return results


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
WEIGHTS_URL = "https://huggingface.co/kornia/dexined/resolve/main/DexiNed_BIPED_10.pth"


def main():
    weights_path = Path("/tmp/dexined_biped.pth")

    if not weights_path.exists():
        print(f"Downloading DexiNed weights from {WEIGHTS_URL}...")
        urllib.request.urlretrieve(WEIGHTS_URL, str(weights_path))
    else:
        print(f"Using cached weights: {weights_path}")

    print("Loading model...")
    model = DexiNed()
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Export to ONNX with fixed 480x640 input (matches OpenCV Zoo convention)
    H, W = 480, 640
    dummy = torch.randn(1, 3, H, W)
    out_path = MODELS_DIR / "dexined.onnx"

    print(f"Exporting to {out_path} with input [{1}, {3}, {H}, {W}]...")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["img"],
        output_names=["out_1", "out_2", "out_3", "out_4", "out_5", "out_6", "block_cat"],
        opset_version=17,
        do_constant_folding=True,
    )

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Exported: {out_path.name} ({size_mb:.1f} MB)")

    # Compress
    import subprocess
    zst_path = out_path.with_suffix(".onnx.zst")
    print(f"Compressing to {zst_path.name}...")
    subprocess.run(["zstd", "-19", "--force", "-q", str(out_path), "-o", str(zst_path)], check=True)
    zst_mb = zst_path.stat().st_size / 1024 / 1024
    print(f"Compressed: {zst_path.name} ({zst_mb:.1f} MB)")

    # Quick verification
    import onnx
    m = onnx.load(str(out_path))
    for inp in m.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"Input: {inp.name} {dims}")
    print(f"Outputs: {len(m.graph.output)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
