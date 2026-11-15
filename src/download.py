"""Fetch the free LOBSTER sample if it is not already in data/."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

# Official samples are listed at https://data.lobsterdata.com/info/DataSamples.php.
# That host did not resolve here, so the bytes come from a public mirror of the
# same AMZN 2012-06-21 level-10 sample. The files stay local and are gitignored.
MESSAGE_NAME = "AMZN_2012-06-21_34200000_57600000_message_10.csv"
BOOK_NAME = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
MIRROR = "https://huggingface.co/datasets/totalorganfailure/lobster-data/resolve/main/LOBSTER_SampleFile_AMZN_2012-06-21_10"


def ensure_sample(data_dir: Path) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    message = data_dir / MESSAGE_NAME
    book = data_dir / BOOK_NAME
    _fetch(f"{MIRROR}/{MESSAGE_NAME}", message)
    _fetch(f"{MIRROR}/{BOOK_NAME}", book)
    return message, book


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "neural-jump-ode-sample"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(dest)
