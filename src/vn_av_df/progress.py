"""Tiến độ dễ theo dõi trên log Kaggle: in theo số lượng và theo thời gian, kèm RAM / GPU.

Progress in mục đầu tiên (biết ngay đã chạy thông, gồm cả thời gian nạp model), rồi mỗi
`every_n` mục hoặc mỗi `every_s` giây; luồng nền in "vẫn chạy" khi lâu không có mục nào
(đoạn khởi động worker / nạp model trước đây im lặng hàng chục phút).
"""

import subprocess
import threading
import time
from pathlib import Path


def ram_free_gb():
    """RAM còn trống (Linux /proc/meminfo); None nếu không đọc được."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def gpu_status():
    """'GPU0 5.1/15.0 GB 87 %' cho mọi GPU (nvidia-smi); rỗng nếu không có GPU."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    parts = []
    for line in out.strip().splitlines():
        try:
            index, used, total, util = (x.strip() for x in line.split(","))
            parts.append(
                f"GPU{index} {float(used) / 1024:.1f}/{float(total) / 1024:.0f} GB {util} %"
            )
        except ValueError:
            continue
    return " · ".join(parts)


def system_status():
    ram = ram_free_gb()
    return " | ".join(
        x for x in (f"RAM trống {ram:.1f} GB" if ram is not None else "", gpu_status()) if x
    )


def duration(seconds):
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} phút"
    return f"{seconds // 3600} giờ {seconds % 3600 // 60:02d} phút"


class Progress:
    """with Progress("avhubert", 7622) as p: ... p.step()  (an toàn khi nhiều luồng gọi)."""

    def __init__(self, label, total, unit="mẫu", every_n=25, every_s=60.0, heartbeat=True):
        self.label, self.total, self.unit = label, total, unit
        self.every_n, self.every_s = every_n, every_s
        self.done = 0
        self.began = self.last = time.monotonic()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._beat, daemon=True) if heartbeat else None

    def __enter__(self):
        if self.thread:
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        return False

    def _beat(self):
        while not self.stop.wait(self.every_s):
            with self.lock:
                quiet = time.monotonic() - self.last
                if quiet >= self.every_s:
                    print(
                        f"{self.label}: vẫn chạy, {self.done}/{self.total} {self.unit} sau "
                        f"{duration(time.monotonic() - self.began)} | {system_status()}",
                        flush=True,
                    )
                    self.last = time.monotonic()

    def step(self, n=1, note=""):
        with self.lock:
            self.done += n
            now = time.monotonic()
            due = (
                self.done == n
                or self.done >= self.total
                or self.done % self.every_n < n
                or now - self.last >= self.every_s
            )
            if not due:
                return
            spent = now - self.began
            rate = self.done / max(spent, 1e-6) * 60
            left = (self.total - self.done) / max(rate, 1e-9) * 60
            first = " (gồm khởi động)" if self.done == n else ""
            print(
                f"{self.label}: {self.done}/{self.total} {self.unit} | {duration(spent)}{first} | "
                f"{rate:.1f} {self.unit}/phút | còn ~{duration(left)} | {system_status()}"
                + (f" | {note}" if note else ""),
                flush=True,
            )
            self.last = now
