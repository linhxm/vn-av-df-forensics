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


def process_ram():
    """'RAM chính 2.1 GB · 4 tiến trình con 18.3 GB' (Linux /proc); rỗng nếu không đọc được.
    Cho biết RAM tụt vì tiến trình chính hay vì worker (part 1: RAM trống 26 → 0,4 GB)."""
    import os

    def rss(pid):
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
        except OSError:
            pass
        return 0.0

    me = os.getpid()
    if not Path(f"/proc/{me}/status").exists():
        return ""
    children = []
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit():
            continue
        try:
            parent = int((folder / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        if parent == me:
            children.append(rss(folder.name))
    out = f"RAM chính {rss(me):.1f} GB"
    if children:
        out += f" · {len(children)} tiến trình con {sum(children):.1f} GB"
    return out


def system_status():
    ram = ram_free_gb()
    parts = (f"RAM trống {ram:.1f} GB" if ram is not None else "", process_ram(), gpu_status())
    return " | ".join(x for x in parts if x)


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
        self.began = self.last = self.moved = time.monotonic()
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
                # Chỉ khi không có mục nào xong trong every_s (khởi động / treo); đang chạy đều thì
                # dòng tiến độ của step() đã đủ.
                if time.monotonic() - self.moved >= self.every_s:
                    print(
                        f"{self.label}: vẫn chạy, {self.done}/{self.total} {self.unit} sau "
                        f"{duration(time.monotonic() - self.began)} | {system_status()}",
                        flush=True,
                    )
                    self.last = time.monotonic()

    def step(self, n=1, note=""):
        with self.lock:
            self.done += n
            now = self.moved = time.monotonic()
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
