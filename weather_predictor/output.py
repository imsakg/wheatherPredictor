"""Protect source datasets and atomically publish local UTF-8 output."""
import os
import tempfile
from pathlib import Path


def check_output_path(source, target):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve() or (target.exists() and os.path.samefile(source,target)):
        raise ValueError("output must not overwrite the input dataset or an alias")


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",newline="",dir=path.parent,delete=False,suffix=".tmp") as stream:
            temporary = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
