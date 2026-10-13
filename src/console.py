"""In ấn trên terminal: tiêu đề bước có dấu [✔], bảng, câu SQL, thông báo vẽ hình, tạm dừng."""
import os
import sys
import time
from contextlib import contextmanager

if os.name == "nt":
    os.system("")  # bật mã màu ANSI trên terminal Windows
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except AttributeError:
        pass

GREEN, RED, YELLOW, CYAN, BOLD, RESET = ("\033[92m", "\033[91m", "\033[93m", "\033[96m",
                                         "\033[1m", "\033[0m")
LINE = "=" * 78
_AUTO = False
_OPEN_FIGS = False
_TOTAL = 0
_DONE = []          # (số bước, tiêu đề, thời gian, trạng thái)
_T0 = time.time()


def setup(auto, open_figs, total):
    global _AUTO, _OPEN_FIGS, _TOTAL, _T0
    _AUTO, _OPEN_FIGS, _TOTAL, _T0 = auto, open_figs, total, time.time()
    _DONE.clear()


@contextmanager
def step(no, title, pause_after=True):
    """Khung một bước:  [ ] Bước i/N ... -> [✔] HOÀN THÀNH (thời gian)  hoặc  [✘] THẤT BẠI."""
    print("\n" + LINE)
    print(f"{BOLD}{CYAN}[ ] BƯỚC {no}/{_TOTAL}. {title}{RESET}")
    print(LINE)
    t0 = time.time()
    try:
        yield
    except SkipStep as s:
        dt = time.time() - t0
        _DONE.append((no, title, dt, "skip"))
        print(f"{YELLOW}[–] BƯỚC {no}/{_TOTAL}. {title} – BỎ QUA: {s}{RESET}")
        return
    except BaseException:
        _DONE.append((no, title, time.time() - t0, "fail"))
        print(f"{RED}[✘] BƯỚC {no}/{_TOTAL}. {title} – THẤT BẠI{RESET}")
        raise
    dt = time.time() - t0
    _DONE.append((no, title, dt, "ok"))
    print(f"{GREEN}[✔] BƯỚC {no}/{_TOTAL}. {title} – HOÀN THÀNH ({dt:.1f}s){RESET}")
    if pause_after:
        pause()


class SkipStep(Exception):
    """Ném ra trong một bước để đánh dấu bước đó là 'bỏ qua' (không phải lỗi)."""


def checklist():
    print(f"\n{BOLD}{'=' * 25} TỔNG KẾT CÁC BƯỚC {'=' * 25}{RESET}")
    mark = {"ok": f"{GREEN}[✔]", "skip": f"{YELLOW}[–]", "fail": f"{RED}[✘]"}
    for no, title, dt, st in _DONE:
        print(f"  {mark[st]}{RESET} Bước {no:>2}: {title} ({dt:.1f}s)")
    print(f"  Tổng thời gian: {elapsed() / 60:.1f} phút")


def sub(title):
    print(f"\n{BOLD}--- {title} {'-' * max(3, 72 - len(title))}{RESET}")


def info(text=""):
    print(f"   {text}")


def ok(text):
    print(f"   {GREEN}✔ {text}{RESET}")


def warn(text):
    print(f"   {YELLOW}! {text}{RESET}")


def err(text):
    print(f"   {RED}✘ {text}{RESET}")


def _fmt(v):
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, int):
        return f"{v:,}".replace(",", ".")
    if isinstance(v, float):
        return f"{v:,.4f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return str(v)


def kv(label, value):
    print(f"   {label:<40s}: {_fmt(value)}")


def table(headers, rows, aligns=None):
    """In bảng từ danh sách Python. Số dùng định dạng Việt Nam: 6.362.620 ; 0,9975."""
    cells = [[_fmt(v) for v in r] for r in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in cells)) if cells else len(str(h))
              for i, h in enumerate(headers)]
    aligns = aligns or ["<"] + [">"] * (len(headers) - 1)
    sep = "   +" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(sep)
    print("   |" + "|".join(f" {h:{a}{w}} " for h, a, w in zip(headers, aligns, widths)) + "|")
    print(sep)
    for r in cells:
        print("   |" + "|".join(f" {v:{a}{w}} " for v, a, w in zip(r, aligns, widths)) + "|")
    print(sep)


def show_sql(text):
    info("Câu lệnh Spark SQL:")
    for line in text.strip().splitlines():
        print(f"      {CYAN}{line.strip()}{RESET}")


def figure_saved(path, source="", values=""):
    print(f"   {GREEN}>> ĐÃ VẼ HÌNH:{RESET} {os.path.relpath(path)}")
    if source:
        print(f"      Số liệu lấy từ : {source}")
    if values:
        print(f"      Giá trị trên hình: {values}")
    if _OPEN_FIGS and os.name == "nt":
        try:
            os.startfile(path)
        except OSError:
            pass


def pause(msg="Nhấn Enter để sang bước tiếp theo..."):
    if not _AUTO:
        try:
            input(f"\n   >>> {msg}")
        except EOFError:
            pass


def elapsed():
    return time.time() - _T0
