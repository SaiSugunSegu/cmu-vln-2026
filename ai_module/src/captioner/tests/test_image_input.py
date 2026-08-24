"""`image_is_complete` decides whether a view is worth sending to a model host.

The bug it exists for: `cv2.imwrite` publishes the path before the bytes, so a reader waiting on
`is_file()` takes the prefix of a 1.4-1.6 MB encode. Measured against a real 1920x640 write, a
polling reader saw an incomplete file 94% of the time. A `livingroom_1` route call built on one
of those prefixes was refused by two independent providers as undecodable, and the question fell
back to a zero-scoring route.

Pure stdlib -- no cv2, no pydantic -- so these run anywhere.
"""
import struct

from captioner.image_input import PNG_SIGNATURE, image_is_complete


def png(width: int = 4, height: int = 4, *, idat_chunks: int = 2) -> bytes:
    """A structurally whole PNG. The pixel data is not real, and does not need to be:
    `image_is_complete` is a truncation check, not a decoder."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + b"\0\0\0\0"

    out = PNG_SIGNATURE
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    for _ in range(idat_chunks):
        out += chunk(b"IDAT", b"\x00" * 512)
    out += chunk(b"IEND", b"")
    return out


def written(tmp_path, data: bytes, name: str = "best_rank1_sofa.png"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


# -- the whole thing -------------------------------------------------------

def test_a_complete_png_passes(tmp_path):
    assert image_is_complete(written(tmp_path, png()))


def test_a_png_with_many_chunks_passes(tmp_path):
    """A 1.6 MB crop is hundreds of IDAT chunks; the walk has to reach IEND through all of
    them, not give up at some fixed count."""
    assert image_is_complete(written(tmp_path, png(idat_chunks=400)))


def test_a_str_path_works_as_well_as_a_path(tmp_path):
    assert image_is_complete(str(written(tmp_path, png())))


# -- every way it can fail -------------------------------------------------

def test_a_truncated_png_fails(tmp_path):
    """The actual bug: a valid header and a cut-off tail, which is what a reader sees while
    cv2 is still encoding."""
    whole = png(idat_chunks=40)
    assert not image_is_complete(written(tmp_path, whole[:len(whole) // 2]))


def test_a_png_cut_one_byte_before_the_end_fails(tmp_path):
    """The tightest case: everything but the last byte of IEND's CRC."""
    whole = png()
    assert not image_is_complete(written(tmp_path, whole[:-1]))


def test_a_signature_with_no_chunks_fails(tmp_path):
    assert not image_is_complete(written(tmp_path, PNG_SIGNATURE))


def test_an_empty_file_fails(tmp_path):
    """cv2 creates the file before it writes anything, so zero bytes is a real state."""
    assert not image_is_complete(written(tmp_path, b""))


def test_a_missing_file_fails(tmp_path):
    assert not image_is_complete(tmp_path / "never_written.png")


def test_a_directory_fails(tmp_path):
    """Reading one raises OSError rather than returning bytes; it must not escape."""
    assert not image_is_complete(tmp_path)


def test_something_that_is_not_a_png_fails(tmp_path):
    """Only PNG is written here, so anything else is a surprise worth refusing rather than
    forwarding to a model host to reject."""
    assert not image_is_complete(written(tmp_path, b"\xff\xd8\xff\xe0" + b"jpeg" * 64))


def test_a_chunk_claiming_more_than_the_file_holds_fails(tmp_path):
    """A length field read from a partial write can point past the end. That must be a False,
    not an IndexError or a hang."""
    body = PNG_SIGNATURE + struct.pack(">I", 1 << 30) + b"IDAT" + b"\0" * 32
    assert not image_is_complete(written(tmp_path, body))


def test_a_png_without_an_iend_fails(tmp_path):
    """Well-formed chunks all the way to the end, but the terminator never arrived."""
    whole = png(idat_chunks=3)
    assert not image_is_complete(written(tmp_path, whole[:whole.rindex(b"IEND") - 4]))
