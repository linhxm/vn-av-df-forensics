"""Controlled temporal baselines on the same frozen FATE descriptors.

These are GRU/TCN/Transformer baselines, not reproductions of AuViRe or FATE's task head.
"""

import math

import torch
from torch import nn

ARCHITECTURES = ("linear", "gru", "tcn", "transformer")


class TemporalDetector(nn.Module):
    """B1 linear hoặc head thời gian; input là một clip [T,D], mask [T]."""

    def __init__(self, audio_dim, visual_dim, architecture="gru", hidden=128, dropout=0.1):
        super().__init__()
        if architecture not in ARCHITECTURES or hidden < 4 or hidden % 4:
            raise ValueError("Unknown architecture or hidden size not divisible by four")
        self.config = dict(
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            architecture=architecture,
            hidden=hidden,
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + visual_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        if architecture == "linear":
            self.fusion = nn.Identity()
            self.classifier = nn.Linear(audio_dim + visual_dim, 1)
            return
        if architecture == "gru":
            self.temporal = nn.GRU(
                hidden, hidden, num_layers=2, dropout=dropout, batch_first=True, bidirectional=True
            )
        elif architecture == "tcn":
            self.temporal = nn.ModuleList(
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for d in (1, 2, 4)
            )
        else:
            self.temporal = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden, 4, hidden * 4, dropout, batch_first=True),
                2,
                enable_nested_tensor=False,
            )
        self.classifier = nn.Linear(hidden * (2 if architecture == "gru" else 1), 1)

    def forward(self, audio, visual, valid):
        """Không cho temporal context đi xuyên qua đoạn thiếu quan sát."""
        if audio.ndim != 2 or visual.shape[0] != len(audio) or valid.shape != (len(audio),):
            raise ValueError("Expected one clip: audio[T,D], visual[T,D], valid[T]")
        if not torch.isfinite(audio[valid]).all() or not torch.isfinite(visual[valid]).all():
            raise ValueError("Nonfinite features")
        x = self.fusion(
            torch.cat(
                [audio.masked_fill(~valid[:, None], 0), visual.masked_fill(~valid[:, None], 0)], -1
            )
        )
        if self.config["architecture"] == "linear":
            return self.classifier(x)[:, 0].masked_fill(~valid, 0)
        # Process contiguous valid runs independently: missing evidence cannot leak through time.
        edges = torch.diff(
            torch.cat([valid.new_tensor([False]), valid, valid.new_tensor([False])]).int()
        )
        result = x.new_zeros(len(x))
        for a, b in zip(torch.where(edges == 1)[0].tolist(), torch.where(edges == -1)[0].tolist()):
            z = x[a:b][None]
            if self.config["architecture"] == "gru":
                z, _ = self.temporal(z)
            elif self.config["architecture"] == "tcn":
                z = z.transpose(1, 2)
                for block in self.temporal:
                    z = z + block(z)
                z = z.transpose(1, 2)
            else:
                pos = torch.arange(b - a, device=x.device)[:, None]
                freq = torch.exp(
                    torch.arange(0, x.shape[-1], 2, device=x.device)
                    * (-math.log(10000) / x.shape[-1])
                )
                pe = torch.zeros_like(z[0])
                pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
                z = self.temporal(z + pe)
            result[a:b] = self.classifier(z)[0, :, 0]
        return result


def pool_score(scores, valid, top_fraction=0.1):
    """Lấy trung bình top-k ô hợp lệ; không có quan sát thì trả None."""
    values = scores[valid]
    if not len(values):
        return None
    return values.topk(max(1, math.ceil(len(values) * top_fraction))).values.mean()


def build_model(
    audio_dim,
    visual_dim,
    architecture="gru",
    hidden=128,
    dropout=0.1,
    native_stride=1,
    artifact_dim=None,
):
    """Khôi phục đúng kiến trúc từ checkpoint, không đổi model âm thầm."""
    from vn_av_df.syncartifact import ARCHITECTURES as SYNC_ARTIFACT

    if architecture in SYNC_ARTIFACT:
        from vn_av_df.syncartifact import SyncArtifactDetector

        if artifact_dim is None:
            raise ValueError("P2 needs artifact features")
        return SyncArtifactDetector(
            audio_dim, visual_dim, architecture, hidden, dropout, native_stride, artifact_dim
        )
    if architecture in {"realrecon", "realrecon_concat", "visual_tcn"}:
        from vn_av_df.reconstruction import RealReconDetector

        return RealReconDetector(
            audio_dim, visual_dim, architecture, hidden, dropout, native_stride
        )
    return TemporalDetector(audio_dim, visual_dim, architecture, hidden, dropout)


def output_valid(model, valid):
    """Mask đầu ra có thể thưa hơn input native 25 Hz của P1."""
    return model.output_valid(valid) if hasattr(model, "output_valid") else valid
