"""Encoder/decoder for LEGEND detector IDs.

Implements the 32-bit integer encoding defined in the `LEGEND data format
specification <https://legend-exp.github.io/legend-data-format-specs/dev/detector_ids/>`_.

Binary format (nibbles): ``RT XX XX XY``

- ``R`` (1 nibble): reserved, always zero
- ``T`` (1 nibble): detector type
- ``XXXXX`` (5 nibbles): serial number
- ``Y`` (1 nibble): sub-serial number
"""

from __future__ import annotations

import operator
import re

__all__ = ["decode_detectorid", "encode_detectorid"]

# Type nibble values
_TYPE_C = 0x1  # Coax HPGe
_TYPE_B = 0x2  # BEGe HPGe
_TYPE_P = 0x3  # PPC HPGe
_TYPE_V = 0x4  # ICPC HPGe
_TYPE_S = 0x9  # SiPM
_TYPE_PMT = 0xA  # PMT
_TYPE_PULS = 0xB  # Pulser
_TYPE_AUX = 0xC  # Auxiliary
_TYPE_DUMMY = 0xD  # Dummy
_TYPE_BSLN = 0xE  # Baseline
_TYPE_MUON = 0xF  # Muon veto

_HPGE_CHAR_TO_TYPE = {"C": _TYPE_C, "B": _TYPE_B, "P": _TYPE_P, "V": _TYPE_V}
_HPGE_TYPE_TO_CHAR = {v: k for k, v in _HPGE_CHAR_TO_TYPE.items()}

# Sub-serial nibble maps to letter: 0=A, 1=B, ..., 15=P
_SUB_SERIAL_LETTERS = "ABCDEFGHIJKLMNOP"
_LETTER_TO_SUB = {c: i for i, c in enumerate(_SUB_SERIAL_LETTERS)}

# Pre-compiled regex patterns for encoding
_RE_C00ANG = re.compile(r"C00ANG([0-9])$")
_RE_C000RG = re.compile(r"C000RG([0-9])$")
_RE_HPGE = re.compile(r"([CBPV])([0-9]{5})([A-P])$")
_RE_SIPM = re.compile(r"S([0-9]{3})$")
_RE_PMT = re.compile(r"PMT([0-9]{3})$")
_RE_PULS = re.compile(r"PULS([0-9]{2})(ANA)?$")
_RE_AUX = re.compile(r"AUX([0-9]{2})$")
_RE_DUMMY = re.compile(r"DUMMY([0-9]{1,2})$")
_RE_BSLN = re.compile(r"BSLN([0-9]{2})$")
_RE_MUON = re.compile(r"MUON([0-9]{2})$")


def encode_detectorid(det_id: str) -> int:
    """Encode a LEGEND detector ID string to its 32-bit integer representation.

    Parameters
    ----------
    det_id
        Detector ID string following the LEGEND data format specification.

    Returns
    -------
    int
        32-bit unsigned integer encoding of the detector ID.

    Raises
    ------
    TypeError
        If `det_id` is not a string.
    ValueError
        If `det_id` does not match any valid detector ID format.

    Examples
    --------
    >>> encode_detectorid("B00000C")
    33554434
    >>> hex(encode_detectorid("B00000C"))
    '0x2000002'
    >>> hex(encode_detectorid("PULS00ANA"))
    '0xb000001'
    """
    if not isinstance(det_id, str):
        msg = f"detector ID must be a string, got {type(det_id).__name__!r}"
        raise TypeError(msg)

    # Special cases for C: C00ANGn and C000RGn must be checked before general HPGe
    m = _RE_C00ANG.match(det_id)
    if m:
        n = int(m.group(1))
        serial = 0xF1000 | n
        return (_TYPE_C << 24) | (serial << 4)

    m = _RE_C000RG.match(det_id)
    if m:
        n = int(m.group(1))
        serial = 0xF2000 | n
        return (_TYPE_C << 24) | (serial << 4)

    # General HPGe: [CBPV]nnnnn[A-P]
    m = _RE_HPGE.match(det_id)
    if m:
        t = _HPGE_CHAR_TO_TYPE[m.group(1)]
        serial = int(m.group(2))
        y = _LETTER_TO_SUB[m.group(3)]
        # Serial is exactly 5 decimal digits, so max is 99999
        return (t << 24) | (serial << 4) | y

    # SiPM: Snnn
    m = _RE_SIPM.match(det_id)
    if m:
        serial = int(m.group(1))
        return (_TYPE_S << 24) | (serial << 4)

    # PMT: PMTnnn
    m = _RE_PMT.match(det_id)
    if m:
        serial = int(m.group(1))
        return (_TYPE_PMT << 24) | (serial << 4)

    # Pulser: PULSnn or PULSnnANA
    m = _RE_PULS.match(det_id)
    if m:
        serial = int(m.group(1))
        y = 1 if m.group(2) is not None else 0
        return (_TYPE_PULS << 24) | (serial << 4) | y

    # Auxiliary: AUXnn
    m = _RE_AUX.match(det_id)
    if m:
        serial = int(m.group(1))
        return (_TYPE_AUX << 24) | (serial << 4)

    # Dummy: DUMMYn (legacy single digit) or DUMMYnn
    m = _RE_DUMMY.match(det_id)
    if m:
        serial = int(m.group(1))
        return (_TYPE_DUMMY << 24) | (serial << 4)

    # Baseline: BSLNnn
    m = _RE_BSLN.match(det_id)
    if m:
        serial = int(m.group(1))
        return (_TYPE_BSLN << 24) | (serial << 4)

    # Muon veto: MUONnn
    m = _RE_MUON.match(det_id)
    if m:
        serial = int(m.group(1))
        return (_TYPE_MUON << 24) | (serial << 4)

    msg = f"invalid detector ID string: {det_id!r}"
    raise ValueError(msg)


def decode_detectorid(value: int) -> str:
    """Decode a 32-bit integer to its LEGEND detector ID string representation.

    Parameters
    ----------
    value
        32-bit unsigned integer encoding of the detector ID.

    Returns
    -------
    str
        Detector ID string following the LEGEND data format specification.

    Raises
    ------
    TypeError
        If `value` is not an integer type.
    ValueError
        If `value` does not represent a valid detector ID.

    Examples
    --------
    >>> decode_detectorid(0x02000002)
    'B00000C'
    >>> decode_detectorid(0x0b000001)
    'PULS00ANA'
    """
    try:
        value = operator.index(value)
    except TypeError:
        msg = f"detector ID must be an integer, got {type(value).__name__!r}"
        raise TypeError(msg) from None

    if value < 0 or value > 0xFFFFFFFF:
        msg = f"detector ID integer out of 32-bit range: {value}"
        raise ValueError(msg)

    r = (value >> 28) & 0xF
    t = (value >> 24) & 0xF
    serial = (value >> 4) & 0xFFFFF
    y = value & 0xF

    if r != 0:
        msg = f"reserved nibble must be zero, got {r:#x} in {value:#010x}"
        raise ValueError(msg)

    if t == 0x0 or t in {0x5, 0x6, 0x7, 0x8}:
        msg = f"reserved detector type nibble: {t:#x}"
        raise ValueError(msg)

    # HPGe types
    if t in _HPGE_TYPE_TO_CHAR:
        char = _HPGE_TYPE_TO_CHAR[t]

        # Special cases only apply to type C (Coax)
        if t == _TYPE_C:
            upper = serial >> 4  # top 4 nibbles of the 5-nibble serial
            n = serial & 0xF  # last nibble

            if upper == 0xF100:
                if y != 0:
                    msg = f"sub-serial must be 0 for C00ANGn encoding, got {y}"
                    raise ValueError(msg)
                if n > 9:
                    msg = f"digit in C00ANGn must be 0-9, got {n}"
                    raise ValueError(msg)
                return f"C00ANG{n}"

            if upper == 0xF200:
                if y != 0:
                    msg = f"sub-serial must be 0 for C000RGn encoding, got {y}"
                    raise ValueError(msg)
                if n > 9:
                    msg = f"digit in C000RGn must be 0-9, got {n}"
                    raise ValueError(msg)
                return f"C000RG{n}"

        if serial > 99999:
            msg = f"HPGe serial number out of range (max 99999): {serial}"
            raise ValueError(msg)
        return f"{char}{serial:05d}{_SUB_SERIAL_LETTERS[y]}"

    if t == _TYPE_S:
        if y != 0:
            msg = f"SiPM sub-serial must be 0, got {y}"
            raise ValueError(msg)
        if serial > 999:
            msg = f"SiPM serial number out of range (max 999): {serial}"
            raise ValueError(msg)
        return f"S{serial:03d}"

    if t == _TYPE_PMT:
        if y != 0:
            msg = f"PMT sub-serial must be 0, got {y}"
            raise ValueError(msg)
        if serial > 999:
            msg = f"PMT serial number out of range (max 999): {serial}"
            raise ValueError(msg)
        return f"PMT{serial:03d}"

    if t == _TYPE_PULS:
        if y > 1:
            msg = f"Pulser sub-serial must be 0 or 1, got {y}"
            raise ValueError(msg)
        if serial > 99:
            msg = f"Pulser serial number out of range (max 99): {serial}"
            raise ValueError(msg)
        suffix = "ANA" if y == 1 else ""
        return f"PULS{serial:02d}{suffix}"

    if t == _TYPE_AUX:
        if y != 0:
            msg = f"AUX sub-serial must be 0, got {y}"
            raise ValueError(msg)
        if serial > 99:
            msg = f"AUX serial number out of range (max 99): {serial}"
            raise ValueError(msg)
        return f"AUX{serial:02d}"

    if t == _TYPE_DUMMY:
        if y != 0:
            msg = f"DUMMY sub-serial must be 0, got {y}"
            raise ValueError(msg)
        if serial > 99:
            msg = f"DUMMY serial number out of range (max 99): {serial}"
            raise ValueError(msg)
        return f"DUMMY{serial:02d}"

    if t == _TYPE_BSLN:
        if y != 0:
            msg = f"BSLN sub-serial must be 0, got {y}"
            raise ValueError(msg)
        if serial > 99:
            msg = f"BSLN serial number out of range (max 99): {serial}"
            raise ValueError(msg)
        return f"BSLN{serial:02d}"

    if t == _TYPE_MUON:
        if y != 0:
            msg = f"MUON sub-serial must be 0, got {y}"
            raise ValueError(msg)
        if serial > 99:
            msg = f"MUON serial number out of range (max 99): {serial}"
            raise ValueError(msg)
        return f"MUON{serial:02d}"

    # Should be unreachable given the reserved-type check above
    msg = f"unknown detector type nibble: {t:#x}"
    raise ValueError(msg)
