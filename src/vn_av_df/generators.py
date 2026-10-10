"""Registry generator: tên dữ liệu phải khớp adapter và checkpoint thực tế."""


def generator_adapter(name):
    """Không tự thay generator khi thiếu dependency hoặc weight."""
    if name == "wav2lip_gan":
        from vn_av_df import wav2lip

        return wav2lip
    if name == "musetalk_1_5":
        from vn_av_df import musetalk

        return musetalk
    raise ValueError(f"Unsupported generator: {name}")


def setup_generators(cfg):
    """Chuẩn bị generator của phiên (generation_generators) hoặc mọi generator trong plan."""
    mapping = cfg["generation"].get("generators_by_split", {"train": ["wav2lip_gan"]})
    names = cfg.get("generation_generators") or {g for names in mapping.values() for g in names}
    return {name: generator_adapter(name).setup(cfg) for name in sorted(names)}
