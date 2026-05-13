# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""I/O for CASCADE annotation files.

`local` — load/save annotation JSONs from a directory on disk.
`hf` — load annotations from a HuggingFace dataset repo (requires the
       optional `hf` extra: `pip install cascade_av[hf]`).
"""

from cascade_av.io.local import (
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
