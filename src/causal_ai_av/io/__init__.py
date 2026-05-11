"""I/O for AV Causal annotation files.

`local` — load/save annotation JSONs from a directory on disk.
`hf` — load annotations from a HuggingFace dataset repo (requires the
       optional `hf` extra: `pip install causal_ai_av[hf]`).
"""

from causal_ai_av.io.local import (
    group_by_clip_id,
    iter_dir,
    load_dir,
    load_file,
    parse_filename,
    save_file,
)

__all__ = [
    "load_file",
    "load_dir",
    "iter_dir",
    "save_file",
    "parse_filename",
    "group_by_clip_id",
]
