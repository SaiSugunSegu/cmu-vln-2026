"""Is this file a whole image yet?

A best-view crop is 1.4-1.6 MB of PNG, and `cv2.imwrite` publishes the path before the bytes.
Anything that waits for the file to *exist* can therefore read it halfway through being encoded,
and what comes back is a valid header followed by nothing. That is not hypothetical: a
`livingroom_1` route call was refused by two independent providers as undecodable, on files that
verify clean on disk after the run.

`best_view._write_image` now publishes atomically, so this should never fire. It stays as the
check on the other side of that contract -- cheap enough to run before every send, and the
difference between a legible "we skipped rank 1, it was still being written" and a 400 from a
model host.

Deliberately not a decode. The signature and the chunk walk are a few microseconds and need no
cv2, so this imports in the reasoners, in the backends and under pytest on a bare host alike.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Union

__all__ = ["PNG_SIGNATURE", "image_is_complete"]

#: The 8 bytes every PNG starts with.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: Chunk header (4-byte length + 4-byte type) plus the trailing 4-byte CRC.
_CHUNK_OVERHEAD = 12


def image_is_complete(path: Union[str, Path]) -> bool:
    """Whether `path` holds a fully written PNG.

    False for missing, empty, truncated, or not a PNG at all -- every way the file can fail to
    be something worth sending, without distinguishing between them, because the caller's
    response to all of them is the same: do not send this one.

    Chunk CRCs are NOT verified. A partial write is truncation, not corruption, and reaching a
    terminating `IEND` means every byte before it arrived. Checking the CRCs would cost a pass
    over 1.6 MB to catch a failure mode that does not occur here.
    """
    try:
        data = Path(path).read_bytes()
    except OSError:
        return False

    if len(data) < len(PNG_SIGNATURE) + _CHUNK_OVERHEAD:
        return False
    if not data.startswith(PNG_SIGNATURE):
        return False

    pos = len(PNG_SIGNATURE)
    end = len(data)
    while pos + _CHUNK_OVERHEAD <= end:
        length = struct.unpack_from(">I", data, pos)[0]
        kind = data[pos + 4:pos + 8]
        nxt = pos + _CHUNK_OVERHEAD + length
        if nxt > end:
            return False                 # the last chunk was cut short mid-write
        if kind == b"IEND":
            return True
        pos = nxt
    return False
