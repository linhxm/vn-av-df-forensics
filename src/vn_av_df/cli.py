"""Lệnh của pipeline v003 (gọi qua generation/0x_*.py, training/0x_*.py hoặc notebook).

setup [--generators ...] [--encoders] [--models]
                                         tải source / weight generator, encoder, tài sản model
plan                                     kế hoạch sinh cho part sạch đang chọn
voice                                    phiên giọng (①②③ + track B)
lipsync [--generators ...]               phiên lip-sync (④⑤⑥) cho các generator chọn
finalize                                 gom record → manifest
prepare [--encoders ...]                 trích feature (mặc định: encoder các model cần)
benchmark [--models ...] [--steps N] [--synthetic]
                                         đo thời gian / bộ nhớ train từng model, ước giờ GPU
train [--models ...] [--seeds ...]       train model (chạy lại thì tiếp từ epoch đã xong)
evaluate [--force]                       test một lần
report                                   bảng kết quả + cổng
"""

import argparse
import json


def model_assets(cfg):
    """Tài sản để train: XLS-R gốc (BL-A fine-tune) + checkpoint AVH-Align công bố (BL-M, M2)."""
    from vn_av_df.assets import fetch_file, hf_snapshot

    x, assets = cfg["encoders"]["xlsr"], cfg.get("model_assets") or {}
    result = {
        "xlsr": hf_snapshot(
            x["hf_id"],
            x["model"],
            ["config.json", "*.bin", "*.safetensors", "preprocessor_config.json"],
        )
    }
    if assets.get("avh_align_init"):
        result["avh_align_init"] = fetch_file(assets["avh_align_url"], assets["avh_align_init"])
    return result


def setup(cfg, generators=None, encoders=False, models=False):
    from vn_av_df.generation.generate import adapter_for

    result = {}
    names = (
        generators
        if generators is not None
        else sorted({g for split in cfg["generation"]["generators"].values() for g in split})
    )
    for name in names:
        adapter = adapter_for(name)
        result[name] = adapter.setup(cfg)
    if encoders:
        from vn_av_df.assets import setup_avhubert

        result["avhubert"] = setup_avhubert(cfg["encoders"]["avhubert"])
    if encoders or models:
        result.update(model_assets(cfg))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vn-av-df")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("setup")
    p.add_argument("--generators", nargs="*")
    p.add_argument("--encoders", action="store_true")
    p.add_argument("--models", action="store_true")
    sub.add_parser("plan")
    sub.add_parser("voice")
    sub.add_parser("lipsync").add_argument("--generators", nargs="*")
    sub.add_parser("finalize")
    sub.add_parser("prepare").add_argument("--encoders", nargs="*")
    p = sub.add_parser("benchmark")
    p.add_argument("--models", nargs="*")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--synthetic", action="store_true")
    p = sub.add_parser("train")
    p.add_argument("--models", nargs="*")
    p.add_argument("--seeds", nargs="*", type=int)
    sub.add_parser("evaluate").add_argument("--force", action="store_true")
    sub.add_parser("report")
    args = parser.parse_args(argv)

    from settings import config

    cfg = config()
    if args.command == "setup":
        result = setup(cfg, args.generators, args.encoders, args.models)
    elif args.command == "plan":
        from vn_av_df.generation.plan import make_plan

        result = make_plan(cfg["clean_dataset"], cfg["plan"], cfg["generation"], cfg["seed"])
    elif args.command in ("voice", "lipsync"):
        from vn_av_df.generation.generate import run_generation

        if args.command == "lipsync" and args.generators:
            cfg["session_generators"] = args.generators
        result = run_generation(cfg, args.command)
    elif args.command == "finalize":
        from vn_av_df.generation.generate import finalize

        result = finalize(cfg)
    elif args.command == "prepare":
        from vn_av_df.training.prepare import encoders_for, prepare

        result = prepare(cfg, args.encoders or encoders_for(cfg["models"]))
    elif args.command == "benchmark":
        from vn_av_df.training.benchmark import benchmark

        result = benchmark(cfg, args.models, args.steps, synthetic=args.synthetic)
    elif args.command == "train":
        from vn_av_df.training.train import train_all

        if args.models:
            cfg["models"] = args.models
        if args.seeds:
            cfg["seeds"] = args.seeds
        result = train_all(cfg)
    elif args.command == "evaluate":
        from vn_av_df.training.evaluate import evaluate_all

        result = evaluate_all(cfg, args.force)
    else:
        from vn_av_df.training.report import build_report

        result = build_report(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0
