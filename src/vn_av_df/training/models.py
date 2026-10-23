"""Model v003: 4 baseline cài lại đúng code gốc + 2 model đề xuất (+ 3 ablation), cùng một giao diện.

Mỗi model:
- needs: cache cần nạp (⊆ {"av", "ssl", "mouth", "pose", "lipinc", "wave"});
- heads: {tên đầu ra: tác vụ} - tác vụ ∈ ai | visual | audio | mismatch; forward trả logit /
  điểm theo frame [B, T] cho từng đầu (điểm cao = nghi ngờ); đầu "score" = một điểm cho mọi tác vụ;
- pool: cách gộp điểm frame thành điểm video ("topk" 10 %, "topk5" 5 frame, "lse", "mean", "max");
- frame_level: False khi model chỉ cho điểm cả video (LIPINC-V2) → bỏ các phép đo định vị;
- train_options: ghi đè cfg["training"] cho đúng công thức train của paper gốc (optimizer, lr,
  batch, epoch, patience, scheduler); real_only: chỉ học từ clip real; pseudo_fake: xác suất thay
  đoạn tiếng khi nạp clip real (data.replace_audio); rawboost: augment RawBoost cho waveform;
- loss(batch, out): mặc định BCE theo frame + BCE video cho từng đầu có nhãn.

Baseline (docs/v003_model.md §4; code gốc đã đối chiếu, khác biệt ghi trong docstring từng lớp):
- lipinc_v2 (BL-V, Datta et al. 2025, github.com/skrantidatta/LIPINC-V2, port từ TensorFlow);
- ssl_aasist (BL-A, Tak et al. 2022, github.com/TakHemlata/SSL_Anti-spoofing, fine-tune XLS-R);
- avh_align (BL-M, Smeu et al. CVPR 2025, github.com/bit-ml/AVH-Align, chỉ học real) và
  avh_align_sup (BL-M/sup, cùng paper, học có nhãn);
- BL-Ens = max(BL-V, BL-A) sau chuẩn hoá theo real validation: không train, tính ở evaluate.py.
Đề xuất: trihead_v2 (M1) + 3 ablation, avh_align_plus (M2).
speechforensics (không train) chỉ giữ cho demo khi chưa có checkpoint.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

MAX_LAG = 15


# ---------------------------------------------------------------- tiện ích chung
def topk_pool(scores, mask, fraction=0.1, k=None):
    """Trung bình top-k điểm hợp lệ mỗi clip (k cố định, hoặc theo tỉ lệ); [B, T] → [B]."""
    out = []
    for s, m in zip(scores, mask):
        v = s[m]
        if not len(v):
            out.append(s.new_tensor(0.0))
            continue
        count = min(len(v), k) if k else max(1, math.ceil(len(v) * fraction))
        out.append(v.topk(count).values.mean())
    return torch.stack(out)


def pool_scores(scores, mask, how):
    if how == "lse":
        return torch.logsumexp(scores.masked_fill(~mask, -1e4), dim=1)
    if how == "mean":
        return (scores * mask).sum(1) / mask.sum(1).clamp(min=1)
    if how == "max":
        return scores.masked_fill(~mask, -1e4).amax(1)
    if how == "topk5":
        return topk_pool(scores.sigmoid(), mask, k=5)
    return topk_pool(scores.sigmoid(), mask)


def masked_bce(logits, target, mask, pos_weight=None):
    if not mask.any():
        return logits.sum() * 0
    weight = None
    if pos_weight is not None:
        weight = torch.where(target > 0.5, pos_weight, torch.ones_like(target))[mask]
    return F.binary_cross_entropy_with_logits(logits[mask], target[mask], weight=weight)


class TemporalBlock(nn.Module):
    """Conv1d giãn có residual trên [B, T, C] (mask: frame ngoài clip bị đặt 0)."""

    def __init__(self, channels, dilations=(1, 2, 4), dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=d, dilation=d),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for d in dilations
        )

    def forward(self, x, mask):
        z = (x * mask[..., None]).transpose(1, 2)
        for block in self.blocks:
            z = z + block(z)
        return z.transpose(1, 2) * mask[..., None]


def shift(x, delta):
    """y[:, t] = x[:, t + delta] (đệm 0 ngoài biên); x [B, T, ...]."""
    if delta == 0:
        return x
    out = torch.zeros_like(x)
    if delta > 0:
        out[:, :-delta] = x[:, delta:]
    else:
        out[:, -delta:] = x[:, :delta]
    return out


class GradReverse(torch.autograd.Function):
    """Đi tới giữ nguyên; đi lùi nhân −λ (Ganin & Lempitsky 2015, DANN)."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


def masked_mean(h, mask):
    m = mask.float()[..., None]
    return (h * m).sum(1) / m.sum(1).clamp(min=1)


class Model(nn.Module):
    needs = {"av"}
    heads = {"score": "ai"}
    pool = "topk"
    trainable = True
    references = 0
    crop_frames = None
    frame_level = True
    real_only = False
    pseudo_fake = 0.0
    rawboost = False
    train_options = {}

    def rebuild_assets(self):
        """Tài sản cần để dựng lại đúng kiến trúc khi nạp checkpoint (lưu cùng state)."""
        return {}

    def on_epoch(self, epoch, total):
        pass

    def video_pool(self, logits, mask):
        k = 5 if self.pool == "topk5" else None
        return topk_pool(logits.sigmoid(), mask, k=k).clamp(1e-6, 1 - 1e-6)

    def loss(self, batch, out, pos_weight=None):
        total = 0
        mask = batch["mask"]
        self.loss_parts = {}  # từng thành phần (frame / video theo tác vụ) để ghi log
        for head, task in self.heads.items():
            target = batch[f"y_{task}"]
            pw = None if pos_weight is None else pos_weight.get(task)
            frame = masked_bce(out[head], target, mask, pw)
            video = ((target * mask).amax(1) > 0).float()
            clip = F.binary_cross_entropy(self.video_pool(out[head], mask), video)
            self.loss_parts[f"{task}_frame"] = float(frame.detach())
            self.loss_parts[f"{task}_video"] = float(clip.detach())
            total = total + frame + clip
        return total


# ---------------------------------------------------------------- không train (demo)
class SpeechForensics(Model):
    """NeurIPS 2024: cos(a_t, v_t+δ) với δ cố định tốt nhất ±15; không train."""

    heads = {"score": "ai"}
    pool = "mean"
    trainable = False

    def __init__(self, **_):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, batch):
        a, v, mask = F.normalize(batch["a"], dim=-1), F.normalize(batch["v"], dim=-1), batch["mask"]
        best, best_sim = None, None
        for delta in range(-MAX_LAG, MAX_LAG + 1):
            sim = (a * shift(v, delta)).sum(-1) * mask * shift(mask.float(), delta)
            mean = sim.sum(1) / mask.sum(1).clamp(min=1)
            if best is None:
                best, best_sim = mean, sim
            else:
                better = mean > best
                best = torch.where(better, mean, best)
                best_sim = torch.where(better[:, None], sim, best_sim)
        return {"score": -best_sim}


# ---------------------------------------------------------------- BL-M · AVH-Align
class AlignMLP(nn.Module):
    """`FusionModel` của AVH-Align (model.py gốc): chiếu riêng hình / tiếng 1024 → 512, ghép, MLP
    1024 → 512 → 256 → 128 → 1 (LayerNorm + ReLU). Tên tham số giữ đúng bản gốc để nạp checkpoint
    `AVH-Align_AV1M.pt` (train trên 45 k video real AV-1M)."""

    def __init__(self, dim=1024, hidden=1024):
        super().__init__()
        self.visual_proj = nn.Linear(dim, hidden // 2)
        self.audio_proj = nn.Linear(dim, hidden // 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, v, a):
        return self.mlp(torch.cat([self.visual_proj(v), self.audio_proj(a)], -1))[..., 0]

    def lag_volume(self, v, a):
        """Φ(v_t, a_t+δ) cho δ ∈ [−15, 15] → [B, T, 31]; tiếng ngoài clip = vector 0 (như gốc)."""
        pv, pa = self.visual_proj(v), self.audio_proj(a)
        zero = self.audio_proj.bias.expand_as(pa)  # audio_proj(0) = bias
        outs = []
        for d in range(-MAX_LAG, MAX_LAG + 1):
            shifted = shift(pa - zero, d) + zero
            outs.append(self.mlp(torch.cat([pv, shifted], -1))[..., 0])
        return torch.stack(outs, -1)


def load_official_align(phi, path):
    """Nạp trọng số `AVH-Align_AV1M.pt` (state_dict của FusionModel) vào AlignMLP."""
    import argparse
    from pathlib import Path

    if not Path(path).is_file():
        raise FileNotFoundError(f"Thiếu checkpoint AVH-Align {path}: chạy training/00_setup.py")
    torch.serialization.add_safe_globals([argparse.Namespace])
    state = torch.load(path, map_location="cpu", weights_only=True)
    phi.load_state_dict(state.get("state_dict", state))


def unit(x):
    """Chuẩn hoá L2 từng frame (gốc chia thẳng cho chuẩn; frame toàn 0 giữ 0)."""
    return x / torch.linalg.norm(x, ord=2, dim=-1, keepdim=True).clamp(min=1e-12)


class AVHAlign(Model):
    """BL-M: AVH-Align không giám sát (train.py gốc), chỉ học từ clip real.

    Loss: với mỗi frame hình t, softmax trên 31 frame tiếng t−15…t+15 (tiếng ngoài clip = 0) và lấy
    −log p tại t (InfoNCE). Điểm frame = −Φ(v_t, a_t); điểm video = log-sum-exp (eval.py gốc).
    Công thức train gốc: Adam lr 1e-5, ReduceLROnPlateau (×0,1, patience 5), dừng sớm 10 epoch,
    tối đa 500 epoch, batch 512 frame (checkpoint công bố). Khác: batch theo clip (8 clip ≈ 1.100
    frame), loss lấy trung bình thay vì tổng (Adam bất biến tỉ lệ), bỏ frame mất mặt (cache có
    mask); khởi tạo từ checkpoint công bố khi có (`assets["avh_align_init"]`).
    """

    pool = "lse"
    real_only = True
    train_options = {
        "optimizer": "adam",
        "lr": 1e-5,
        "weight_decay": 0.0,
        "batch_size": 8,
        "epochs": 500,
        "patience": 10,
        "scheduler": "plateau",
        "plateau_patience": 5,
        "clip": None,
    }

    def __init__(self, dim, assets=None, **_):
        super().__init__()
        self.heads = {"score": "ai"}
        self.phi = AlignMLP(dim)
        assets = assets or {}
        if assets.get("load_pretrained", True) and assets.get("avh_align_init"):
            load_official_align(self.phi, assets["avh_align_init"])

    def forward(self, batch):
        a, v = unit(batch["a"]), unit(batch["v"])
        return {"score": -self.phi(v, a), "_a": a, "_v": v}

    def loss(self, batch, out, pos_weight=None):
        mask = batch["mask"]
        logits = self.phi.lag_volume(out["_v"], out["_a"])  # [B, T, 31]
        nll = -logits.log_softmax(-1)[..., MAX_LAG]
        self.loss_parts = {"infonce": 0.0}
        if not mask.any():
            return logits.sum() * 0
        value = nll[mask].mean()
        self.loss_parts["infonce"] = float(value.detach())
        return value


class AVHAlignSup(AVHAlign):
    """BL-M/sup: AVH-Align/sup (avh_sup/mlp.py gốc, model_type mlp, input both): cùng Φ, điểm frame
    = Φ(v_t, a_t), điểm video s = log-sum-exp theo frame, loss = cross-entropy trên (−s, s) với nhãn
    real / fake của video. Gốc: Adam lr 1e-3, batch 1 video, 100 epoch, dừng sớm 10. Nhãn = AI?
    (label). Khởi tạo ngẫu nhiên như gốc. Augment dịch tiếng −1…+2 theo protocol v003."""

    real_only = False
    train_options = {
        "optimizer": "adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 1,
        "epochs": 100,
        "patience": 10,
        "clip": None,
    }

    def __init__(self, dim, **_):
        Model.__init__(self)
        self.heads = {"score": "ai"}
        self.phi = AlignMLP(dim)

    def forward(self, batch):
        a, v = unit(batch["a"]), unit(batch["v"])
        return {"score": self.phi(v, a)}

    def loss(self, batch, out, pos_weight=None):
        s = torch.logsumexp(out["score"].masked_fill(~batch["mask"], -1e4), dim=1)
        target = batch["video"]["ai"].long()
        value = F.cross_entropy(torch.stack([-s, s], 1), target)
        self.loss_parts = {"ce": float(value.detach())}
        return value


# ---------------------------------------------------------------- BL-V · LIPINC-V2
class KerasMHA(nn.Module):
    """tf.keras.layers.MultiHeadAttention(num_heads, key_dim=d_model): mỗi head d_model chiều."""

    def __init__(self, d_model, heads):
        super().__init__()
        self.heads, self.key = heads, d_model
        self.q = nn.Linear(d_model, heads * d_model)
        self.k = nn.Linear(d_model, heads * d_model)
        self.v = nn.Linear(d_model, heads * d_model)
        self.o = nn.Linear(heads * d_model, d_model)

    def forward(self, x):
        B, N, _ = x.shape

        def split(t):
            return t.view(B, N, self.heads, self.key).transpose(1, 2)

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        w = (q @ k.transpose(-1, -2) / math.sqrt(self.key)).softmax(-1)
        return self.o((w @ v).transpose(1, 2).reshape(B, N, -1))


class KerasEncoderLayer(nn.Module):
    """Một lớp encoder kiểu post-norm của bản gốc: LN(x + MHA(x)) → LN(x + FFN(x))."""

    def __init__(self, d_model, heads):
        super().__init__()
        self.mha = KerasMHA(d_model, heads)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-3)
        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-3)

    def forward(self, x):
        x = self.norm1(x + self.mha(x))
        return self.norm2(x + self.ff2(F.relu(self.ff1(x))))


class VisionTemporalTransformer(nn.Module):
    """MSTIE của LIPINC-V2: patch 8×8 → Dense 128 + vị trí → 2 lớp spatial → TB theo patch → 2 lớp
    temporal → TB theo frame → Dense 192 (ReLU) → bản đồ 8×8×3."""

    def __init__(self, height=64, width=144, patch=8, d_model=128, heads=4, layers=2):
        super().__init__()
        self.patch = patch
        patches = (height // patch) * (width // patch)
        self.projection = nn.Linear(patch * patch * 3, d_model)
        self.position = nn.Parameter(torch.randn(1, patches, d_model))
        self.spatial = nn.ModuleList(KerasEncoderLayer(d_model, heads) for _ in range(layers))
        self.temporal = nn.ModuleList(KerasEncoderLayer(d_model, heads) for _ in range(layers))
        self.out = nn.Linear(d_model, 8 * 8 * 3)

    def patches(self, x):
        """[B, F, H, W, 3] → [B·F, số patch, 8·8·3] theo thứ tự hàng của tf.image.extract_patches."""
        B, Fr, H, W, C = x.shape
        p = self.patch
        x = x.reshape(B * Fr, H // p, p, W // p, p, C).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(B * Fr, (H // p) * (W // p), p * p * C)

    def forward(self, x):
        B, Fr = x.shape[:2]
        first = self.projection(
            self.patches(x)
        )  # "lớp đầu của MSTIE" (dùng cho inconsistency loss)
        h = first + self.position
        for layer in self.spatial:
            h = layer(h)
        h = h.view(B, Fr, -1, h.shape[-1]).mean(2)
        for layer in self.temporal:
            h = layer(h)
        maps = F.relu(self.out(h.mean(1))).view(B, 8, 8, 3)
        return maps, first.view(B, Fr, -1)


def keras_attention(q, k, v, d_k=192):
    """MultiHeadAttention tự viết của LIPINC-V2: softmax(Q Kᵀ / √192) V trên 2 trục cuối [B,8,8,C]."""
    return (q @ k.transpose(-1, -2) / math.sqrt(d_k)).softmax(-1) @ v


def global_ssim(x, y, c1=(0.01 * 2) ** 2, c2=(0.03 * 2) ** 2):
    """SSIM toàn cục (một cửa sổ) giữa hai tập vector [..., N], khả vi."""
    mx, my = x.mean(-1), y.mean(-1)
    vx, vy = x.var(-1, unbiased=False), y.var(-1, unbiased=False)
    cov = ((x - mx[..., None]) * (y - my[..., None])).mean(-1)
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))


class LipincV2(Model):
    """BL-V: LIPINC-V2 (model.py gốc, Keras), port sang PyTorch đúng kích thước từng lớp.

    Đầu vào: 8 crop miệng màu 64×144 (5 frame liên tiếp + 3 frame cùng tư thế, encoders.LipincEncoder)
    và 7 ảnh hiệu |R_t+1 − R_t|. Hai MSTIE (RGB / hiệu) → 3 lớp attention chéo → CNN (64-64-128-128,
    BatchNorm, MaxPool 2 / 4) → Dense 1024 → 1024 → 2 (lớp 1 = real, như demo.py gốc).
    Loss: cross-entropy + 5 × inconsistency loss (paper: SSIM giữa mọi cặp frame trên lớp đầu của
    MSTIE, trung bình AvgS, BCE với nhãn real = 1). Bản công bố tính SSIM bằng numpy (không có
    gradient); ở đây dùng SSIM toàn cục khả vi trên đặc trưng chiếu patch của nhánh RGB.
    Gốc: Adam lr 1e-3, epsilon 0,1, batch 64, 100 epoch. Khác: ảnh chia 255 (gốc đưa 0-255),
    ảnh hiệu tính bằng số nguyên có dấu (gốc trừ uint8 nên bị tràn), nhãn = visual fake.
    Chỉ cho điểm cả video → frame_level = False.
    """

    needs = {"lipinc"}
    pool = "mean"
    frame_level = False
    train_options = {
        "optimizer": "adam",
        "lr": 1e-3,
        "adam_eps": 0.1,
        "weight_decay": 0.0,
        "batch_size": 64,
        "epochs": 100,
        "patience": 10,
        "clip": None,
    }
    inconsistency_weight = 5.0

    def __init__(self, **_):
        super().__init__()
        self.heads = {"score": "visual"}
        self.frame_branch = VisionTemporalTransformer()
        self.residue_branch = VisionTemporalTransformer()

        def block(cin, cout, k):
            return [
                nn.Conv2d(cin, cout, k, padding=k // 2),
                nn.ReLU(),
                nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01),
            ]

        self.cnn = nn.Sequential(
            *block(12, 64, 3),
            *block(64, 64, 1),
            nn.MaxPool2d(2),
            *block(64, 128, 3),
            *block(128, 128, 1),
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
        )

    def forward(self, batch):
        x = batch["lipinc"].float() / 255.0  # [B, 8, 64, 144, 3]
        residue = (x[:, 1:] - x[:, :-1]).abs()
        frame_map, first = self.frame_branch(x)
        residue_map, _ = self.residue_branch(residue)
        color = torch.cat([frame_map, keras_attention(residue_map, frame_map, frame_map)], -1)
        structure = torch.cat(
            [residue_map, keras_attention(frame_map, residue_map, residue_map)], -1
        )
        fusion = keras_attention(structure, color, color)
        logits = self.cnn(torch.cat([structure, fusion], -1).permute(0, 3, 1, 2))
        fake = (logits[:, 0] - logits[:, 1])[:, None].expand(-1, batch["mask"].shape[1])
        return {"score": fake, "_logits": logits, "_first": first}

    def loss(self, batch, out, pos_weight=None):
        real = 1 - batch["video"]["visual"].long()
        ce = F.cross_entropy(out["_logits"], real)
        first = out["_first"]  # [B, 8, patch·128]
        pairs = global_ssim(first[:, :, None], first[:, None, :])  # [B, 8, 8], gồm cả i = j như gốc
        avg = pairs.mean((1, 2)).clamp(1e-4, 1 - 1e-4)
        inconsistency = F.binary_cross_entropy(avg, real.float())
        self.loss_parts = {"ce": float(ce.detach()), "inconsistency": float(inconsistency.detach())}
        return ce + self.inconsistency_weight * inconsistency


# ---------------------------------------------------------------- BL-A · XLS-R + AASIST
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, temperature=1.0):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(nn.init.xavier_normal_(torch.empty(out_dim, 1)))
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x):
        x = self.input_drop(x)
        n = x.size(1)
        att = x.unsqueeze(2).expand(-1, -1, n, -1)
        att = torch.tanh(self.att_proj(att * att.transpose(1, 2)))
        att = F.softmax(torch.matmul(att, self.att_weight) / self.temp, dim=-2)
        x = self.proj_with_att(torch.matmul(att.squeeze(-1), x)) + self.proj_without_att(x)
        size = x.size()
        return self.act(self.bn(x.view(-1, size[-1])).view(size))


class HtrgGraphAttentionLayer(nn.Module):
    """HS-GAL của AASIST: 3 vector attention (cùng loại / chéo) + nút master (stack node)."""

    def __init__(self, in_dim, out_dim, temperature=1.0):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        def param():
            return nn.Parameter(nn.init.xavier_normal_(torch.empty(out_dim, 1)))

        self.att_weight11, self.att_weight22 = param(), param()
        self.att_weight12, self.att_weightM = param(), param()
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x1, x2, master=None):
        n1, n2 = x1.size(1), x2.size(1)
        x = torch.cat([self.proj_type1(x1), self.proj_type2(x2)], dim=1)
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        x = self.input_drop(x)
        n = x.size(1)
        att = x.unsqueeze(2).expand(-1, -1, n, -1)
        att = torch.tanh(self.att_proj(att * att.transpose(1, 2)))
        board = torch.zeros_like(att[..., :1])
        board[:, :n1, :n1] = torch.matmul(att[:, :n1, :n1], self.att_weight11)
        board[:, n1:, n1:] = torch.matmul(att[:, n1:, n1:], self.att_weight22)
        board[:, :n1, n1:] = torch.matmul(att[:, :n1, n1:], self.att_weight12)
        board[:, n1:, :n1] = torch.matmul(att[:, n1:, :n1], self.att_weight12)
        att = F.softmax(board / self.temp, dim=-2)
        att_m = torch.tanh(self.att_projM(x * master))
        att_m = F.softmax(torch.matmul(att_m, self.att_weightM) / self.temp, dim=-2)
        master = self.proj_with_attM(
            torch.matmul(att_m.squeeze(-1).unsqueeze(1), x)
        ) + self.proj_without_attM(master)
        x = self.proj_with_att(torch.matmul(att.squeeze(-1), x)) + self.proj_without_att(x)
        size = x.size()
        x = self.act(self.bn(x.view(-1, size[-1])).view(size))
        return x.narrow(1, 0, n1), x.narrow(1, n1, n2), master


class GraphPool(nn.Module):
    def __init__(self, k, in_dim, p):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        scores = torch.sigmoid(self.proj(self.drop(h)))
        _, n, d = h.size()
        _, idx = torch.topk(scores, max(int(n * self.k), 1), dim=1)
        return torch.gather(h * scores, 1, idx.expand(-1, -1, d))


class ResidualBlock(nn.Module):
    """Residual_block của SSL_Anti-spoofing (giữ cả chi tiết bn1 tính nhưng conv1 nhận x gốc)."""

    def __init__(self, filters, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv1 = nn.Conv2d(filters[0], filters[1], (2, 3), padding=(1, 1))
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.conv2 = nn.Conv2d(filters[1], filters[1], (2, 3), padding=(0, 1))
        self.downsample = filters[0] != filters[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(filters[0], filters[1], (1, 3), padding=(0, 1))

    def forward(self, x):
        identity = x
        if not self.first:
            self.selu(self.bn1(x))  # bản gốc tính rồi bỏ: conv1 dùng x
        out = self.conv2(self.selu(self.bn2(self.conv1(x))))
        if self.downsample:
            identity = self.conv_downsample(identity)
        return out + identity


class AASIST(nn.Module):
    """Back-end AASIST trên đặc trưng SSL (class Model của SSL_Anti-spoofing/model.py)."""

    def __init__(self, ssl_dim=1024):
        super().__init__()
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims, pool_ratios, temps = [64, 32], [0.5, 0.5, 0.5, 0.5], [2.0, 2.0, 100.0, 100.0]
        self.LL = nn.Linear(ssl_dim, 128)
        self.first_bn = nn.BatchNorm2d(1)
        self.first_bn1 = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(
            nn.Sequential(ResidualBlock(filts[1], first=True)),
            nn.Sequential(ResidualBlock(filts[2])),
            nn.Sequential(ResidualBlock(filts[3])),
            nn.Sequential(ResidualBlock(filts[4])),
            nn.Sequential(ResidualBlock(filts[4])),
            nn.Sequential(ResidualBlock(filts[4])),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, 1), nn.SELU(inplace=True), nn.BatchNorm2d(128), nn.Conv2d(128, 64, 1)
        )
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.GAT_layer_S = GraphAttentionLayer(64, gat_dims[0], temps[0])
        self.GAT_layer_T = GraphAttentionLayer(64, gat_dims[0], temps[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temps[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temps[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temps[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temps[2])
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, feat):
        x = self.LL(feat).transpose(1, 2).unsqueeze(1)  # [B, 1, 128, T]
        x = self.selu(self.first_bn(F.max_pool2d(x, (3, 3))))
        x = self.selu(self.first_bn1(self.encoder(x)))
        w = self.attention(x)
        e_S = torch.sum(x * F.softmax(w, dim=-1), dim=-1).transpose(1, 2) + self.pos_S
        out_S = self.pool_S(self.GAT_layer_S(e_S))
        e_T = torch.sum(x * F.softmax(w, dim=-2), dim=-2).transpose(1, 2)
        out_T = self.pool_T(self.GAT_layer_T(e_T))
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1, out_T1 = self.pool_hS1(out_S1), self.pool_hT1(out_T1)
        t_aug, s_aug, m_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1, out_S1, master1 = out_T1 + t_aug, out_S1 + s_aug, master1 + m_aug
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2, out_T2 = self.pool_hS2(out_S2), self.pool_hT2(out_T2)
        t_aug, s_aug, m_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2, out_S2, master2 = out_T2 + t_aug, out_S2 + s_aug, master2 + m_aug
        out_T = torch.max(self.drop_way(out_T1), self.drop_way(out_T2))
        out_S = torch.max(self.drop_way(out_S1), self.drop_way(out_S2))
        master = torch.max(self.drop_way(master1), self.drop_way(master2))
        hidden = torch.cat(
            [
                torch.abs(out_T).amax(1),
                out_T.mean(1),
                torch.abs(out_S).amax(1),
                out_S.mean(1),
                master.squeeze(1),
            ],
            1,
        )
        return self.out_layer(self.drop(hidden))


def xlsr_backbone(spec, pretrained):
    """Wav2Vec2 XLS-R: spec = thư mục HF (from_pretrained) hoặc dict cấu hình (dựng lại / test)."""
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    if isinstance(spec, dict):
        config = Wav2Vec2Config(**spec)
    else:
        config = Wav2Vec2Config.from_pretrained(str(spec))
    # SSL_Anti-spoofing gọi fairseq với mask=False: không che thời gian / kênh khi fine-tune.
    config.apply_spec_augment, config.mask_time_prob, config.mask_feature_prob = False, 0.0, 0.0
    if pretrained and not isinstance(spec, dict):
        return Wav2Vec2Model.from_pretrained(str(spec), config=config)
    return Wav2Vec2Model(config)


SSL_SAMPLES = 64600  # data_utils_SSL.pad: ~4 s


class SSLAASIST(Model):
    """BL-A: XLS-R 300M fine-tune cùng AASIST (Tak et al., Odyssey 2022, SSL_Anti-spoofing).

    Train đúng công thức gốc (script DF): cửa sổ 64.600 mẫu (~4 s, ngắn hơn thì lặp lại), RawBoost
    algo 3, Adam lr 1e-6, weight decay 1e-4, batch 14, cross-entropy có trọng số [0,1 giả; 0,9 thật],
    100 epoch. Tiếng không chuẩn hoá. Lớp 1 = thật như gốc; điểm giả = logit_giả − logit_thật.
    Khác: clip 4-7 s nên khi đánh giá trượt cửa sổ 4 s bước 1 s, điểm frame = max các cửa sổ phủ nó,
    điểm video = max; nhãn cửa sổ = có frame tiếng giả trong cửa sổ; checkpoint chọn theo val loss.
    """

    needs = {"wave"}
    pool = "max"
    crop_frames = 101  # ≥ 64.600 mẫu
    rawboost = True
    train_options = {
        "optimizer": "adam",
        "lr": 1e-6,
        "weight_decay": 1e-4,
        "batch_size": 14,
        "epochs": 100,
        "patience": 10,
        "clip": None,
        "amp": True,
    }
    hop = 16000

    def __init__(self, assets=None, **_):
        super().__init__()
        assets = assets or {}
        self.heads = {"audio": "audio"}
        self.spec = assets.get("xlsr") or {}
        self.ssl = xlsr_backbone(self.spec, assets.get("load_pretrained", True))
        self.backend = AASIST(self.ssl.config.hidden_size)

    def rebuild_assets(self):
        return {"xlsr": self.ssl.config.to_dict()}

    def score_windows(self, wave):
        feat = self.ssl(wave).last_hidden_state
        logits = self.backend(feat.float())
        return logits

    @staticmethod
    def tile(x, size):
        if x.shape[-1] >= size:
            return x[..., :size]
        repeats = math.ceil(size / max(1, x.shape[-1]))
        return x.repeat(1, repeats)[..., :size]

    def forward(self, batch):
        wave, mask = batch["wave"], batch["mask"]
        B, T, step = wave.shape
        out = torch.full((B, T), -1e4, device=wave.device)
        logits_all = []
        for i in range(B):
            n = int(mask[i].nonzero().max()) + 1 if mask[i].any() else T
            flat = wave[i, :n].reshape(1, -1)
            if self.training or flat.shape[1] <= SSL_SAMPLES:
                starts = [0]
            else:
                last = flat.shape[1] - SSL_SAMPLES
                starts = sorted(set(list(range(0, last + 1, self.hop)) + [last]))
            windows = torch.cat([self.tile(flat[:, s:], SSL_SAMPLES) for s in starts])
            logits = self.score_windows(windows)
            logits_all.append(logits)
            fake = logits[:, 0] - logits[:, 1]
            for s, value in zip(starts, fake):
                a, b = s // step, min(n, math.ceil((s + SSL_SAMPLES) / step))
                out[i, a:b] = torch.maximum(out[i, a:b], value)
            out[i, :n] = torch.where(out[i, :n] <= -1e4, fake.max(), out[i, :n])
        return {"audio": out, "_logits": logits_all}

    def loss(self, batch, out, pos_weight=None):
        mask = batch["mask"]
        spoof = ((batch["y_audio"] * mask).amax(1) > 0).long()
        logits = torch.cat([chunk[:1] for chunk in out["_logits"]])
        weight = torch.tensor([0.1, 0.9], device=logits.device)
        value = F.cross_entropy(logits.float(), 1 - spoof, weight=weight)
        self.loss_parts = {"wce": float(value.detach())}
        return value


# ---------------------------------------------------------------- M1 · TriHead-v2
class SyncTrunk(nn.Module):
    """Cost volume tiếng-môi theo độ trễ ±15 → CNN 2D (thời gian × độ trễ) → đặc trưng frame."""

    def __init__(self, dim, hidden=128, dropout=0.1):
        super().__init__()
        lags = 2 * MAX_LAG + 1
        self.pa = nn.Sequential(nn.Linear(dim, 256), nn.LayerNorm(256))
        self.pv = nn.Sequential(nn.Linear(dim, 256), nn.LayerNorm(256))
        self.cost = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=(2, 1), dilation=(2, 1)),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=(4, 1), dilation=(4, 1)),
            nn.GELU(),
        )
        self.lag_score = nn.Conv2d(32, 1, 1)
        self.content = nn.Sequential(nn.Linear(2 * dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.mix = nn.Sequential(nn.Linear(67 + hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.temporal = TemporalBlock(hidden, dropout=dropout)
        self.register_buffer("lags", torch.arange(-MAX_LAG, MAX_LAG + 1).float())
        self.n_lags = lags

    def volume(self, a, v, mask):
        cos_a, cos_v = F.normalize(a, dim=-1), F.normalize(v, dim=-1)
        pa, pv = self.pa(a), self.pv(v)
        rng = range(-MAX_LAG, MAX_LAG + 1)
        learned = torch.stack([(pa * shift(pv, d)).sum(-1) / 16.0 for d in rng], -1)
        cosine = torch.stack([(cos_a * shift(cos_v, d)).sum(-1) for d in rng], -1)
        valid = torch.stack([shift(mask, d) for d in rng], -1) & mask[..., None]
        return torch.stack([learned, cosine], 1) * valid[:, None], valid  # [B, 2, T, L]

    def forward(self, a, v, mask):
        volume, valid = self.volume(a, v, mask)
        h = self.cost(volume)
        score = self.lag_score(h)[:, 0].masked_fill(~valid, -1e4)
        weights = score.softmax(-1)
        lag = (weights * self.lags).sum(-1, keepdim=True) / MAX_LAG
        peak = score.amax(-1, keepdim=True).clamp(-20, 20) / 10
        entropy = -(weights * weights.clamp(min=1e-8).log()).sum(-1, keepdim=True) / math.log(
            self.n_lags
        )
        pooled = torch.cat([h.mean(-1), h.amax(-1)], 1).transpose(1, 2)
        feature = torch.cat([pooled, lag, peak, entropy], -1)
        h = self.mix(torch.cat([feature, self.content(torch.cat([a, v], -1))], -1))
        return self.temporal(h, mask.float())


def lip_kinematics(lips):
    """[B, T, 20, 2] → y và 3 bậc sai phân [B, T, 80]."""
    y = lips[..., 1]
    orders = [y]
    for _ in range(3):
        prev = orders[-1]
        orders.append(torch.cat([torch.zeros_like(prev[:, :1]), prev[:, 1:] - prev[:, :-1]], 1))
    return torch.cat(orders, -1)


def velocity(x):
    return torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], 1)


def trace_streams(mouth):
    """[B, T, H, W] uint8 → [B, 3, T, H, W]: ảnh, NPR, residual thời gian.

    NPR đúng Tan et al. (CVPR 2024): hiệu mỗi pixel với pixel tham chiếu (góc trên-trái) của ô 2×2,
    tức x − up_nearest(x[::2, ::2]) (bản trước dùng trung bình 2×2 = residual làm mờ).
    """
    x = mouth.float() / 255.0
    ref = x[..., ::2, ::2]
    up = ref.repeat_interleave(2, -2).repeat_interleave(2, -1)[..., : x.shape[-2], : x.shape[-1]]
    prev = torch.cat([x[:, :1], x[:, :-1]], 1)
    nxt = torch.cat([x[:, 1:], x[:, -1:]], 1)
    return torch.stack([x - 0.5, (x - up) * 4, (x - (prev + nxt) / 2) * 4], 1)


class TraceTrunk(nn.Module):
    """Nhánh miệng: 3 luồng residual → Conv3D → pool → ghép động học môi + tư thế đầu → TCN."""

    def __init__(self, hidden=128, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 24, 3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(24),
            nn.GELU(),
            nn.Conv3d(24, 48, 3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(48),
            nn.GELU(),
            nn.Conv3d(48, 64, 3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
        )
        # y + vận tốc + gia tốc môi (60) + tư thế đầu và vận tốc của nó (6): lệch môi–đầu (LipDA).
        self.kin = nn.Sequential(nn.Linear(66, 64), nn.GELU())
        self.mix = nn.Sequential(nn.Linear(128 + 64, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.temporal = TemporalBlock(hidden, dropout=dropout)

    def forward(self, mouth, lips, pose, mask):
        h = self.cnn(trace_streams(mouth))
        feature = torch.cat([h.mean((-1, -2)), h.amax((-1, -2))], 1).transpose(1, 2)
        kin = torch.cat([lip_kinematics(lips)[..., :60], pose, velocity(pose)], -1)
        feature = torch.cat([feature, self.kin(kin)], -1)
        return self.temporal(self.mix(feature), mask.float())


class AudioTrunk(nn.Module):
    """XLS-R (đóng băng, cache theo nhóm lớp) → trọng số softmax học được trên các nhóm → chiếu →
    2 lớp self-attention theo thời gian (AASIST-lite)."""

    def __init__(self, hidden=128, dropout=0.1, groups=3, dim=1024):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(groups))
        self.proj = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        layer = nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, dropout, batch_first=True)
        self.attention = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)

    def forward(self, ssl, mask):
        groups = ssl.shape[2]
        weights = self.layer_logits[:groups].softmax(0)
        mixed = (ssl * weights[None, None, :, None]).sum(2)
        return self.attention(self.proj(mixed), src_key_padding_mask=~mask)


class FiLM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gamma_beta = nn.Linear(hidden, 2 * hidden)
        nn.init.zeros_(self.gamma_beta.weight)
        nn.init.zeros_(self.gamma_beta.bias)

    def forward(self, h, ref):
        gamma, beta = self.gamma_beta(ref).chunk(2, -1)
        return torch.cat([h * (1 + gamma[:, None]) + beta[:, None], h - ref[:, None]], -1)


class TriHeadV2(Model):
    """M1: 3 nhánh (sync A1, trace, audio) → 3 đầu (visual / audio / mismatch) + tham chiếu người.

    So với TriHead cũ, gộp các sửa P1 trên board kiến trúc (docs/v003_model.md §5):
    NPR đúng paper; luồng tư thế đầu; trọng số học được trên nhóm lớp XLS-R; gộp điểm top-5 frame
    (đoạn giả ngắn nhất 0,3 s ≈ 7 frame); pseudo-fake thay đoạn tiếng khi nạp (p = 0,25);
    gradient reversal lên đầu phân loại người của nhánh audio và trace (λ tăng 0 → 0,3).
    """

    pool = "topk5"
    crop_frames = 75
    pseudo_fake = 0.25
    grl_max = 0.3

    def __init__(
        self,
        dim,
        hidden=128,
        dropout=0.1,
        use_ref=True,
        use_trace=True,
        use_sync=True,
        assets=None,
        **_,
    ):
        super().__init__()
        assets = assets or {}
        self.use = {"ref": use_ref, "audio": True, "trace": use_trace, "sync": use_sync}
        self.needs = {"ssl"} | ({"av"} if use_sync else set()) | ({"mouth"} if use_trace else set())
        self.references = 3 if use_ref else 0
        self.n_speakers = int(assets.get("n_speakers") or 0)
        self.grl = 0.0
        self.heads = {}
        if use_sync:
            self.sync = SyncTrunk(dim, hidden, dropout)
        if use_trace:
            self.trace = TraceTrunk(hidden, dropout)
        self.audio = AudioTrunk(hidden, dropout)
        width = 2 * hidden if use_ref else hidden
        if use_ref:
            self.film = nn.ModuleDict(
                {k: FiLM(hidden) for k in ("sync", "trace", "audio") if self.use[k]}
            )
        visual_in = width * (int(use_trace) + int(use_sync))
        self.visual_head = nn.Sequential(
            nn.Linear(visual_in, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.audio_head = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 1))
        mismatch_in = width * (int(use_sync) + 1)
        self.mismatch_head = nn.Sequential(
            nn.Linear(mismatch_in, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.heads = {"visual": "visual", "audio": "audio", "mismatch": "mismatch"}
        if self.n_speakers:
            self.speaker_heads = nn.ModuleDict(
                {k: nn.Linear(hidden, self.n_speakers) for k in ("audio", "trace") if self.use[k]}
            )

    def rebuild_assets(self):
        return {"n_speakers": self.n_speakers}

    def on_epoch(self, epoch, total):
        p = epoch / max(1, total)
        self.grl = self.grl_max * (2 / (1 + math.exp(-10 * p)) - 1)

    def branches(self, batch):
        mask = batch["mask"]
        out = {"audio": self.audio(batch["ssl"], mask)}
        if self.use["sync"]:
            out["sync"] = self.sync(batch["a"], batch["v"], mask)
        if self.use["trace"]:
            out["trace"] = self.trace(batch["mouth"], batch["lips"], batch["pose"], mask)
        return out

    def reference(self, batch):
        """Trung bình (theo thời gian rồi theo các clip tham chiếu) đặc trưng nhánh; không grad."""
        refs = batch["refs"]
        B, R, T = refs["valid"].shape
        flat = {k: v.flatten(0, 1) for k, v in refs.items()}
        flat["mask"] = flat.pop("valid")
        with torch.no_grad():
            feats = self.branches(flat)
        count = batch["ref_count"].clamp(min=1)
        out = {}
        for name, h in feats.items():
            per_clip = masked_mean(h, flat["mask"]).view(B, R, -1)
            present = (torch.arange(R, device=h.device)[None] < batch["ref_count"][:, None]).float()
            out[name] = (per_clip * present[..., None]).sum(1) / count[:, None]
        return out

    def forward(self, batch):
        feats = self.branches(batch)
        out = {}
        if self.n_speakers and self.training:
            for k, head in self.speaker_heads.items():
                pooled = masked_mean(feats[k], batch["mask"])
                out[f"_speaker_{k}"] = head(GradReverse.apply(pooled, self.grl))
        if self.use["ref"]:
            if "refs" in batch and batch["ref_count"].max() > 0:
                refs = self.reference(batch)
            else:
                refs = {k: torch.zeros_like(h[:, 0]) for k, h in feats.items()}
            feats = {k: self.film[k](h, refs[k]) for k, h in feats.items()}
        out["visual"] = self.visual_head(
            torch.cat([feats[k] for k in ("trace", "sync") if k in feats], -1)
        )[..., 0]
        out["audio"] = self.audio_head(feats["audio"])[..., 0]
        out["mismatch"] = self.mismatch_head(
            torch.cat([feats[k] for k in ("sync", "audio") if k in feats], -1)
        )[..., 0]
        return out

    def loss(self, batch, out, pos_weight=None):
        total = super().loss(batch, out, pos_weight)
        speaker = batch.get("speaker")
        if speaker is not None and (speaker >= 0).any():
            keep = speaker >= 0
            for k in ("audio", "trace"):
                if f"_speaker_{k}" in out:
                    ce = F.cross_entropy(out[f"_speaker_{k}"][keep], speaker[keep])
                    self.loss_parts[f"speaker_{k}"] = float(ce.detach())
                    total = total + ce
        return total


# ---------------------------------------------------------------- M2 · AVH-Align+
class AVHAlignPlus(Model):
    """M2: nâng cấp trực tiếp AVH-Align trên cùng nền (Φ = AlignMLP khởi tạo từ checkpoint công
    bố). Thêm: (1) Φ chấm trên cả dải độ trễ ±15 → cost volume → CNN 2D + soft-argmin (lõi A1);
    (2) luồng môi–đầu (vận tốc môi + tư thế đầu); (3) pseudo-fake thay đoạn tiếng. Học có giám
    sát trên cùng dữ liệu với BL-M/sup → so trực tiếp. Đầu ra: ai, mismatch. Φ học với lr 1e-5
    (như AVH-Align), phần mới với lr chung."""

    needs = {"av", "pose"}
    pool = "topk5"
    pseudo_fake = 0.25
    phi_lr = 1e-5

    def __init__(self, dim, hidden=128, dropout=0.1, assets=None, **_):
        super().__init__()
        assets = assets or {}
        self.heads = {"ai": "ai", "mismatch": "mismatch"}
        self.phi = AlignMLP(dim)
        if assets.get("load_pretrained", True) and assets.get("avh_align_init"):
            load_official_align(self.phi, assets["avh_align_init"])
        self.cost = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=(2, 1), dilation=(2, 1)),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=(4, 1), dilation=(4, 1)),
            nn.GELU(),
        )
        self.lag_score = nn.Conv2d(32, 1, 1)
        self.register_buffer("lags", torch.arange(-MAX_LAG, MAX_LAG + 1).float())
        self.motion = nn.Sequential(nn.Linear(20 + 6, 32), nn.GELU())
        self.mix = nn.Sequential(nn.Linear(67 + 32, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.temporal = TemporalBlock(hidden, dropout=dropout)
        self.out = nn.Linear(hidden, 2)

    def optimizer_groups(self, lr):
        phi = list(self.phi.parameters())
        ids = {id(p) for p in phi}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return [{"params": phi, "lr": self.phi_lr}, {"params": rest, "lr": lr}]

    def forward(self, batch):
        mask = batch["mask"]
        a, v = unit(batch["a"]), unit(batch["v"])
        volume = self.phi.lag_volume(v, a)  # [B, T, 31]: Φ(v_t, a_t+δ)
        lag_pos = (self.lags / MAX_LAG).expand_as(volume)
        valid = torch.stack([shift(mask, d) for d in range(-MAX_LAG, MAX_LAG + 1)], -1)
        valid = valid & mask[..., None]
        h = self.cost(torch.stack([volume * valid, lag_pos], 1))
        score = self.lag_score(h)[:, 0].masked_fill(~valid, -1e4)
        weights = score.softmax(-1)
        lag = (weights * self.lags).sum(-1, keepdim=True) / MAX_LAG
        peak = score.amax(-1, keepdim=True).clamp(-20, 20) / 10
        entropy = -(weights * weights.clamp(min=1e-8).log()).sum(-1, keepdim=True) / math.log(31)
        pooled = torch.cat([h.mean(-1), h.amax(-1)], 1).transpose(1, 2)
        lips_v = velocity(batch["lips"][..., 1])
        pose = batch["pose"]
        motion = self.motion(torch.cat([lips_v, pose, velocity(pose)], -1))
        feature = self.mix(torch.cat([pooled, lag, peak, entropy, motion], -1))
        logits = self.out(self.temporal(feature, mask.float()))
        return {"ai": logits[..., 0], "mismatch": logits[..., 1]}


# ---------------------------------------------------------------- registry
SPECS = {
    "speechforensics": (SpeechForensics, {}),
    "lipinc_v2": (LipincV2, {}),
    "ssl_aasist": (SSLAASIST, {}),
    "avh_align": (AVHAlign, {}),
    "avh_align_sup": (AVHAlignSup, {}),
    "trihead_v2": (TriHeadV2, {}),
    "trihead_v2_no_sync": (TriHeadV2, {"use_sync": False}),
    "trihead_v2_no_trace": (TriHeadV2, {"use_trace": False}),
    "trihead_v2_no_ref": (TriHeadV2, {"use_ref": False}),
    "avh_align_plus": (AVHAlignPlus, {}),
}


def build(name, dim=1024, hidden=128, dropout=0.1, assets=None):
    cls, options = SPECS[name]
    model = cls(dim=dim, hidden=hidden, dropout=dropout, assets=assets, **options)
    model.name = name
    model.augment = "mouth" in model.needs
    both_sides = "av" in model.needs or {"ssl", "mouth"} <= set(model.needs)
    # Dịch tiếng so với hình khi train (data.lag_audio) cho mọi model có giám sát dùng cả hai phía;
    # avh_align không giám sát học "độ lệch 0 = cặp đúng" từ real nên không dịch.
    model.lag_augment = both_sides and model.trainable and not model.real_only
    return model
