from __future__ import annotations

import logging
from dataclasses import dataclass

import numba
import numpy as np
from numpy import int16, int32, ubyte, uint16, uint32
from numpy.typing import NDArray

from .. import types as lgdo
from ..utils import numba_defaults_kwargs as nb_kwargs
from .base import WaveformCodec

log = logging.getLogger(__name__)

# fmt: off
_radware_sigcompress_mask = uint16([0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023,
                                   2047, 4095, 8191, 16383, 32767, 65535])
# fmt: on


@dataclass(frozen=True)
class RadwareSigcompress(WaveformCodec):
    """`radware-sigcompress` array codec.

    Examples
    --------
    >>> from lgdo.compression import RadwareSigcompress
    >>> codec = RadwareSigcompress(codec_shift=-32768)
    """

    codec_shift: int = 0
    """Offset added to the input waveform before encoding.

    The `radware-sigcompress` algorithm is limited to encoding of 16-bit
    integer values. In certain cases (notably, with *unsigned* 16-bit integer
    values), shifting incompatible data by a fixed amount circumvents the
    issue.
    """


def encode(
    sig_in: NDArray | lgdo.VectorOfVectors | lgdo.ArrayOfEqualSizedArrays,
    sig_out: NDArray[ubyte] = None,
    shift: int32 = 0,
) -> (
    (NDArray[ubyte], NDArray[uint32])
    | lgdo.VectorOfEncodedVectors
    | lgdo.ArrayOfEncodedEqualSizedArrays
):
    """Compress digital signal(s) with `radware-sigcompress`.

    Wraps :func:`._radware_sigcompress_encode` and adds support for encoding
    LGDO arrays. Resizes the encoded array to its actual length.

    Note
    ----
    If `sig_in` is a NumPy array, no resizing of `sig_out` is performed. Not
    even of the internally allocated one.

    Because of the current (hardware vectorized) implementation, providing a
    pre-allocated :class:`.VectorOfEncodedVectors` or
    :class:`.ArrayOfEncodedEqualSizedArrays` as `sig_out` is not possible.

    Note
    ----
    The compression algorithm internally interprets the input waveform values as
    16-bit integers. Make sure that your signal can be safely cast to such a
    numeric type. If not, you may want to apply a `shift` to the waveform.

    Parameters
    ----------
    sig_in
        array(s) holding the input signal(s).
    sig_out
        pre-allocated unsigned 8-bit integer array(s) for the compressed
        signal(s). If not provided, a new one will be allocated.
    shift
        value to be added to `sig_in` before compression.

    Returns
    -------
    sig_out, nbytes | LGDO
        given pre-allocated `sig_out` structure or new structure of unsigned
        8-bit integers, plus the number of bytes (length) of the encoded
        signal. If `sig_in` is an :class:`.LGDO`, only a newly allocated
        :class:`.VectorOfEncodedVectors` or
        :class:`.ArrayOfEncodedEqualSizedArrays` is returned.

    See Also
    --------
    ._radware_sigcompress_encode
    """
    if isinstance(sig_in, np.ndarray):
        s = sig_in.shape
        if len(sig_in) == 0:
            return np.empty(s[:-1] + (0,), dtype=ubyte), np.empty(0, dtype=uint32)

        if sig_out is None:
            # the encoded signal is an array of bytes
            # -> twice as long as a uint16
            # pre-allocate ubyte (uint8) array, expand last dimension
            sig_out = np.empty(s[:-1] + (s[-1] * 2,), dtype=ubyte)

        if sig_out.dtype != ubyte:
            msg = "sig_out must be of type ubyte"
            raise ValueError(msg)

        # nbytes has one dimension less (the last one)
        nbytes = np.empty(s[:-1], dtype=uint32)
        # shift too, but let the user specify one value for all waveforms
        # and give it the right shape
        if not hasattr(shift, "__len__"):
            shift = np.full(s[:-1], shift, dtype=int32)

        _radware_sigcompress_encode(
            sig_in, sig_out, shift, nbytes, _radware_sigcompress_mask
        )

        # return without resizing
        return sig_out, nbytes

    if isinstance(sig_in, lgdo.VectorOfVectors):
        if sig_out is not None:
            log.warning(
                "a pre-allocated VectorOfEncodedVectors was given "
                "to hold an encoded ArrayOfEqualSizedArrays. "
                "This is not supported at the moment, so a new one "
                "will be allocated to replace it"
            )
        # convert VectorOfVectors to ArrayOfEqualSizedArrays so it can be
        # directly passed to the low-level encoding routine
        sig_out_nda, nbytes = encode(
            sig_in.to_aoesa(fill_val=0, preserve_dtype=True), shift=shift
        )

        # build the encoded LGDO
        encoded_data = lgdo.ArrayOfEqualSizedArrays(nda=sig_out_nda).to_vov(
            cumulative_length=np.cumsum(nbytes, dtype=uint32)
        )
        # decoded_size is an array, compute it by diff'ing the original VOV
        decoded_size = np.diff(sig_in.cumulative_length, prepend=uint32(0))

        return lgdo.VectorOfEncodedVectors(encoded_data, decoded_size)

    if isinstance(sig_in, lgdo.ArrayOfEqualSizedArrays):
        if sig_out is not None:
            log.warning(
                "a pre-allocated ArrayOfEncodedEqualSizedArrays was given "
                "to hold an encoded ArrayOfEqualSizedArrays. "
                "This is not supported at the moment, so a new one "
                "will be allocated to replace it"
            )

        # encode the internal numpy array
        sig_out_nda, nbytes = encode(sig_in.nda, shift=shift)

        # build the encoded LGDO
        encoded_data = lgdo.ArrayOfEqualSizedArrays(nda=sig_out_nda).to_vov(
            cumulative_length=np.cumsum(nbytes, dtype=uint32)
        )
        return lgdo.ArrayOfEncodedEqualSizedArrays(
            encoded_data, decoded_size=sig_in.nda.shape[1]
        )

    if isinstance(sig_in, lgdo.Array):
        # encode the internal numpy array
        sig_out_nda, nbytes = encode(sig_in.nda, sig_out, shift=shift)
        return lgdo.Array(sig_out_nda), nbytes

    msg = f"unsupported input signal type ({type(sig_in)})"
    raise ValueError(msg)


def decode(
    sig_in: NDArray[ubyte]
    | lgdo.VectorOfEncodedVectors
    | lgdo.ArrayOfEncodedEqualSizedArrays,
    sig_out: NDArray | lgdo.ArrayOfEqualSizedArrays = None,
    shift: int32 = 0,
) -> (NDArray, NDArray[uint32]) | lgdo.VectorOfVectors | lgdo.ArrayOfEqualSizedArrays:
    """Decompress digital signal(s) with `radware-sigcompress`.

    Wraps :func:`._radware_sigcompress_decode` and adds support for decoding
    LGDOs. Resizes the decoded signals to their actual length.

    Note
    ----
    If `sig_in` is a NumPy array, no resizing (along the last dimension) of
    `sig_out` to its actual length is performed. Not even of the internally
    allocated one. If a pre-allocated :class:`.ArrayOfEqualSizedArrays` is
    provided, it won't be resized too. The internally allocated
    :class:`.ArrayOfEqualSizedArrays` `sig_out` has instead always the correct
    size.

    Because of the current (hardware vectorized) implementation, providing a
    pre-allocated :class:`.VectorOfVectors` as `sig_out` is not possible.

    Parameters
    ----------
    sig_in
        array(s) holding the input, compressed signal(s). Output of
        :func:`.encode`.
    sig_out
        pre-allocated array(s) for the decompressed signal(s).  If not
        provided, will allocate a 32-bit integer array(s) structure.
    shift
        the value the original signal(s) was shifted before compression.  The
        value is *subtracted* from samples in `sig_out` right after decoding.

    Returns
    -------
    sig_out, nbytes | LGDO
        given pre-allocated structure or new structure of 32-bit integers, plus
        the number of bytes (length) of the decoded signal.

    See Also
    --------
    ._radware_sigcompress_decode
    """
    # expect the output of encode()
    if isinstance(sig_in, tuple):
        s = sig_in[0].shape
        if sig_out is None:
            # allocate output array with lasd dim as large as the longest
            # uncompressed wf
            maxs = np.max(_get_hton_u16(sig_in[0], 0))
            sig_out = np.empty(s[:-1] + (maxs,), dtype=int32)

        # siglen has one dimension less (the last)
        siglen = np.empty(s[:-1], dtype=uint32)

        if len(sig_in[0]) == 0:
            return sig_out, siglen

        # call low-level routine
        # does not need to know sig_in[1]
        _radware_sigcompress_decode(
            sig_in[0], sig_out, shift, siglen, _radware_sigcompress_mask
        )

        return sig_out, siglen

    if isinstance(sig_in, lgdo.ArrayOfEncodedEqualSizedArrays):
        if sig_out is None:
            # initialize output structure with decoded_size
            sig_out = lgdo.ArrayOfEqualSizedArrays(
                dims=(1, 1),
                shape=(len(sig_in), sig_in.decoded_size.value),
                dtype=int32,
                attrs=sig_in.getattrs(),
            )

        siglen = np.empty(len(sig_in), dtype=uint32)
        # save original encoded vector lengths
        nbytes = np.diff(sig_in.encoded_data.cumulative_length.nda, prepend=uint32(0))

        if len(sig_in) == 0:
            return sig_out

        # convert vector of vectors to array of equal sized arrays
        # can now decode on the 2D matrix together with number of bytes to read per row
        _, siglen = decode(
            (sig_in.encoded_data.to_aoesa(fill_val=0, preserve_dtype=True).nda, nbytes),
            sig_out if isinstance(sig_out, np.ndarray) else sig_out.nda,
            shift=shift,
        )

        # sanity check
        assert np.all(sig_in.decoded_size.value == siglen)

        return sig_out

    if isinstance(sig_in, lgdo.VectorOfEncodedVectors):
        if sig_out:
            log.warning(
                "a pre-allocated VectorOfVectors was given "
                "to hold an encoded VectorOfVectors. "
                "This is not supported at the moment, so a new one "
                "will be allocated to replace it"
            )

        siglen = np.empty(len(sig_in), dtype=uint32)
        # save original encoded vector lengths
        nbytes = np.diff(sig_in.encoded_data.cumulative_length.nda, prepend=uint32(0))

        # convert vector of vectors to array of equal sized arrays
        # can now decode on the 2D matrix together with number of bytes to read per row
        sig_out, siglen = decode(
            (sig_in.encoded_data.to_aoesa(fill_val=0, preserve_dtype=True).nda, nbytes),
            shift=shift,
        )

        # sanity check
        assert np.array_equal(sig_in.decoded_size, siglen)

        # converto to VOV before returning
        return sig_out.to_vov(np.cumsum(siglen, dtype=uint32))

    msg = "unsupported input signal type"
    raise ValueError(msg)


@numba.jit(**nb_kwargs(nopython=True))
def _set_hton_u16(a: NDArray[ubyte], i: int, x: int) -> int:
    """Store an unsigned 16-bit integer value in an array of unsigned 8-bit integers.

    The first two most significant bytes from `x` are stored contiguously in
    `a` with big-endian order.
    """
    x_u16 = uint16(x)
    i_1 = i * 2
    i_2 = i_1 + 1
    a[i_1] = ubyte(x_u16 >> 8)
    a[i_2] = ubyte(x_u16 >> 0)
    return x


@numba.jit(**nb_kwargs(nopython=True))
def _get_hton_u16(a: NDArray[ubyte], i: int) -> uint16:
    """Read unsigned 16-bit integer values from an array of unsigned 8-bit integers.

    The first two most significant bytes of the values must be stored
    contiguously in `a` with big-endian order.
    """
    i_1 = i * 2
    i_2 = i_1 + 1
    if a.ndim == 1:
        return uint16(a[i_1] << 8 | a[i_2])

    return a[..., i_1].astype("uint16") << 8 | a[..., i_2]


@numba.jit("uint16(uint32)", **nb_kwargs(nopython=True))
def _get_high_u16(x: uint32) -> uint16:
    return uint16(x >> 16)


@numba.jit("uint32(uint32, uint16)", **nb_kwargs(nopython=True))
def _set_high_u16(x: uint32, y: uint16) -> uint32:
    return uint32(x & 0x0000FFFF | (y << 16))


@numba.jit("uint16(uint32)", **nb_kwargs(nopython=True))
def _get_low_u16(x: uint32) -> uint16:
    return uint16(x >> 0)


@numba.jit("uint32(uint32, uint16)", **nb_kwargs(nopython=True))
def _set_low_u16(x: uint32, y: uint16) -> uint32:
    return uint32(x & 0xFFFF0000 | (y << 0))


@numba.guvectorize(
    [
        "void(uint16[:], byte[:], int32[:], uint32[:], uint16[:])",
        "void(uint32[:], byte[:], int32[:], uint32[:], uint16[:])",
        "void(uint64[:], byte[:], int32[:], uint32[:], uint16[:])",
        "void( int16[:], byte[:], int32[:], uint32[:], uint16[:])",
        "void( int32[:], byte[:], int32[:], uint32[:], uint16[:])",
        "void( int64[:], byte[:], int32[:], uint32[:], uint16[:])",
    ],
    "(n),(m),(),(),(o)",
    **nb_kwargs(nopython=True),
)
def _radware_sigcompress_encode(
    sig_in: NDArray,
    sig_out: NDArray[ubyte],
    shift: int32,
    siglen: uint32,
    _mask: NDArray[uint16] = _radware_sigcompress_mask,
) -> None:
    """Compress a digital signal.

    Shifts the signal values by ``+shift`` and internally interprets the result
    as :any:`numpy.int16`. Shifted signals must be therefore representable as
    :any:`numpy.int16`, for lossless compression.

    Note
    ----
    The algorithm also computes the first derivative of the input signal, which
    cannot always be represented as a 16-bit integer. In such cases, overflows
    occur, but they seem to be innocuous.

    Almost literal translations of ``compress_signal()`` from the
    `radware-sigcompress` v1.0 C-code by David Radford [1]_. Summary of
    changes:

    - Shift the input signal by `shift` before encoding.
    - Store encoded, :class:`numpy.uint16` signal as an array of bytes
      (:class:`numpy.ubyte`), in big-endian ordering.
    - Declare mask globally to avoid extra memory allocation.
    - Enable hardware-vectorization with Numba (:func:`numba.guvectorize`).
    - Add a couple of missing array boundary checks.

    .. [1] `radware-sigcompress source code
       <https://legend-exp.github.io/legend-data-format-specs/dev/data_compression/#radware-sigcompress-1>`_.
       released under MIT license `[Copyright (c) 2018, David C. Radford
       <radforddc@ornl.gov>]`.

    Parameters
    ----------
    sig_in
        array of integers holding the input signal. In the original C code,
        an array of 16-bit integers was expected.
    sig_out
        pre-allocated array for the unsigned 8-bit encoded signal. In the
        original C code, an array of unsigned 16-bit integers was expected.
    shift
        value to be added to `sig_in` before compression.
    siglen
        array that will hold the lengths of the compressed signals.

    Returns
    -------
    length
        number of bytes in the encoded signal
    """
    mask = _mask
    shift = shift[0]

    i = j = max1 = max2 = min1 = min2 = ds = int16(0)
    nb1 = nb2 = iso = nw = bp = dd1 = dd2 = int16(0)
    dd = uint32(0)

    _set_hton_u16(sig_out, iso, sig_in.size)

    iso += 1
    while j < sig_in.size:  # j = starting index of section of signal
        # find optimal method and length for compression
        # of next section of signal
        si_j = int16(sig_in[j] + shift)
        max1 = min1 = si_j
        max2 = int32(-16000)
        min2 = int32(16000)
        nb1 = nb2 = 2
        nw = 1
        i = j + 1
        # FIXME: 48 could be tuned better?
        while (i < sig_in.size) and (i < j + 48):
            si_i = int16(sig_in[i] + shift)
            si_im1 = int16(sig_in[i - 1] + shift)
            if max1 < si_i:
                max1 = si_i
            if min1 > si_i:
                min1 = si_i
            ds = si_i - si_im1
            if max2 < ds:
                max2 = ds
            if min2 > ds:
                min2 = ds
            nw += 1
            i += 1
        if max1 - min1 <= max2 - min2:  # use absolute values
            nb2 = 99
            while (max1 - min1) > mask[nb1]:
                nb1 += 1
            while (i < sig_in.size) and (
                i < j + 128
            ):  # FIXME: 128 could be tuned better?
                si_i = int16(sig_in[i] + shift)
                if max1 < si_i:
                    max1 = si_i
                dd1 = max1 - min1
                if min1 > si_i:
                    dd1 = max1 - si_i
                if dd1 > mask[nb1]:
                    break
                if min1 > si_i:
                    min1 = si_i
                nw += 1
                i += 1
        else:  # use difference values
            nb1 = 99
            while max2 - min2 > mask[nb2]:
                nb2 += 1
            while (i < sig_in.size) and (
                i < j + 128
            ):  # FIXME: 128 could be tuned better?
                si_i = int16(sig_in[i] + shift)
                si_im1 = int16(sig_in[i - 1] + shift)
                ds = si_i - si_im1
                if max2 < ds:
                    max2 = ds
                dd2 = max2 - min2
                if min2 > ds:
                    dd2 = max2 - ds
                if dd2 > mask[nb2]:
                    break
                if min2 > ds:
                    min2 = ds
                nw += 1
                i += 1

        if bp > 0:
            iso += 1
        # do actual compression
        _set_hton_u16(sig_out, iso, nw)
        iso += 1
        bp = 0
        if nb1 <= nb2:
            # encode absolute values
            _set_hton_u16(sig_out, iso, nb1)
            iso += 1
            _set_hton_u16(sig_out, iso, uint16(min1))
            iso += 1

            i = iso
            while i <= (iso + nw * nb1 / 16):
                _set_hton_u16(sig_out, i, 0)
                i += 1

            i = j
            while i < j + nw:
                dd = int16(sig_in[i] + shift) - min1  # value to encode
                dd = dd << (32 - bp - nb1)
                _set_hton_u16(
                    sig_out, iso, _get_hton_u16(sig_out, iso) | _get_high_u16(dd)
                )
                bp += nb1
                if bp > 15:
                    iso += 1
                    _set_hton_u16(sig_out, iso, _get_low_u16(dd))
                    bp -= 16
                i += 1

        else:
            # encode derivative / difference values
            _set_hton_u16(sig_out, iso, nb2 + 32)  # bits used for encoding, plus flag
            iso += 1
            _set_hton_u16(sig_out, iso, int16(si_j))  # starting signal value
            iso += 1
            _set_hton_u16(sig_out, iso, int16(min2))  # min value used for encoding
            iso += 1

            i = iso
            while i <= iso + nw * nb2 / 16:
                _set_hton_u16(sig_out, i, 0)
                i += 1

            i = j + 1
            while i < j + nw:
                si_i = int16(sig_in[i] + shift)
                si_im1 = int16(sig_in[i - 1] + shift)
                dd = si_i - si_im1 - min2
                dd = dd << (32 - bp - nb2)
                _set_hton_u16(
                    sig_out, iso, _get_hton_u16(sig_out, iso) | _get_high_u16(dd)
                )
                bp += nb2
                if bp > 15:
                    iso += 1
                    _set_hton_u16(sig_out, iso, _get_low_u16(dd))
                    bp -= 16
                i += 1
        j += nw

    if bp > 0:
        iso += 1

    if iso % 2 > 0:
        iso += 1

    siglen[0] = 2 * iso  # number of bytes in compressed signal data


@numba.guvectorize(
    [
        "void(byte[:], uint16[:], int32[:], uint32[:], uint16[:])",
        "void(byte[:], uint32[:], int32[:], uint32[:], uint16[:])",
        "void(byte[:], uint64[:], int32[:], uint32[:], uint16[:])",
        "void(byte[:],  int16[:], int32[:], uint32[:], uint16[:])",
        "void(byte[:],  int32[:], int32[:], uint32[:], uint16[:])",
        "void(byte[:],  int64[:], int32[:], uint32[:], uint16[:])",
    ],
    "(n),(m),(),(),(o)",
    **nb_kwargs(nopython=True),
)
def _radware_sigcompress_decode(
    sig_in: NDArray[ubyte],
    sig_out: NDArray,
    shift: int32,
    siglen: uint32,
    _mask: NDArray[uint16] = _radware_sigcompress_mask,
) -> None:
    """Deompress a digital signal.

    After decoding, the signal values are shifted by ``-shift`` to restore the
    original waveform. The dtype of `sig_out` must be large enough to contain it.

    Almost literal translations of ``decompress_signal()`` from the
    `radware-sigcompress` v1.0 C-code by David Radford [1]_. See
    :func:`._radware_sigcompress_encode` for a list of changes to the original
    algorithm.

    Parameters
    ----------
    sig_in
        array holding the input, compressed signal. In the original code, an
        array of 16-bit unsigned integers was expected.
    sig_out
        pre-allocated array for the decompressed signal. In the original code,
        an array of 16-bit integers was expected.
    shift
        the value the original signal(s) was shifted before compression.  The
        value is *subtracted* from samples in `sig_out` right after decoding.

    Returns
    -------
    length
        length of output, decompressed signal.
    """
    mask = _mask
    shift = shift[0]

    i = j = min_val = nb = isi = iso = nw = bp = int16(0)
    dd = uint32(0)

    sig_len_in = int(sig_in.size / 2)
    _siglen = int16(_get_hton_u16(sig_in, isi))  # signal length
    isi += 1

    while (isi < sig_len_in) and (iso < _siglen):
        if bp > 0:
            isi += 1
        bp = 0  # bit pointer
        nw = _get_hton_u16(sig_in, isi)  # number of samples encoded in this chunk
        isi += 1
        nb = _get_hton_u16(sig_in, isi)  # number of bits used in compression
        isi += 1

        if nb < 32:
            # decode absolute values
            min_val = int16(_get_hton_u16(sig_in, isi))  # min value used for encoding
            isi += 1
            dd = _set_low_u16(dd, _get_hton_u16(sig_in, isi))
            i = 0
            while (i < nw) and (iso < _siglen):
                if (bp + nb) > 15:
                    bp -= 16
                    dd = _set_high_u16(dd, _get_hton_u16(sig_in, isi))
                    isi += 1
                    if isi < sig_len_in:
                        dd = _set_low_u16(dd, _get_hton_u16(sig_in, isi))
                    dd = dd << (bp + nb)
                else:
                    dd = dd << nb
                sig_out[iso] = (_get_high_u16(dd) & mask[nb]) + min_val - shift
                iso += 1
                bp += nb
                i += 1
        else:
            nb -= 32
            #  decode derivative / difference values
            sig_out[iso] = (
                int16(_get_hton_u16(sig_in, isi)) - shift
            )  # starting signal value
            iso += 1
            isi += 1
            min_val = int16(_get_hton_u16(sig_in, isi))  # min value used for encoding
            isi += 1
            if isi < sig_len_in:
                dd = _set_low_u16(dd, _get_hton_u16(sig_in, isi))

            i = 1
            while (i < nw) and (iso < _siglen):
                if (bp + nb) > 15:
                    bp -= 16
                    dd = _set_high_u16(dd, _get_hton_u16(sig_in, isi))
                    isi += 1
                    if isi < sig_len_in:
                        dd = _set_low_u16(dd, _get_hton_u16(sig_in, isi))
                    dd = dd << (bp + nb)
                else:
                    dd = dd << nb
                sig_out[iso] = (
                    int16(
                        (_get_high_u16(dd) & mask[nb])
                        + min_val
                        + sig_out[iso - 1]
                        + shift
                    )
                    - shift
                )
                iso += 1
                bp += nb
                i += 1
        j += nw

    if _siglen != iso:
        msg = "failure: unexpected signal length after decompression"
        raise RuntimeError(msg)

    siglen[0] = _siglen  # number of shorts in decompressed signal data
