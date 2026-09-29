"""Tests for lgdo.detectorid encode/decode functions."""

from __future__ import annotations

import pytest

from lgdo.detectorid import decode_detectorid, encode_detectorid

# ---------------------------------------------------------------------------
# Specification test cases (all bidirectional)
# ---------------------------------------------------------------------------

SPEC_CASES = [
    ("B00000C", 0x02000002),
    ("B59231A", 0x020E75F0),
    ("C00000A", 0x01000000),
    ("C83847I", 0x01147878),
    ("C000RG4", 0x01F20040),
    ("C00ANG7", 0x01F10070),
    ("P94752A", 0x03172200),
    ("P00000K", 0x0300000A),
    ("V99999J", 0x041869F9),
    ("V98237P", 0x0417FBDF),
    ("S000", 0x09000000),
    ("S632", 0x09002780),
    ("S999", 0x09003E70),
    ("PMT000", 0x0A000000),
    ("PMT183", 0x0A000B70),
    ("PMT999", 0x0A003E70),
    ("PULS00", 0x0B000000),
    ("PULS00ANA", 0x0B000001),
    ("PULS99", 0x0B000630),
    ("PULS99ANA", 0x0B000631),
    ("AUX00", 0x0C000000),
    ("AUX99", 0x0C000630),
    ("DUMMY00", 0x0D000000),
    ("DUMMY09", 0x0D000090),
    ("DUMMY10", 0x0D0000A0),
    ("DUMMY99", 0x0D000630),
    ("BSLN00", 0x0E000000),
    ("BSLN99", 0x0E000630),
    ("MUON00", 0x0F000000),
    ("MUON99", 0x0F000630),
]


@pytest.mark.parametrize(("det_str", "det_int"), SPEC_CASES)
def test_encode_spec_cases(det_str, det_int):
    assert encode_detectorid(det_str) == det_int


@pytest.mark.parametrize(("det_str", "det_int"), SPEC_CASES)
def test_decode_spec_cases(det_str, det_int):
    assert decode_detectorid(det_int) == det_str


@pytest.mark.parametrize(("det_str", "det_int"), SPEC_CASES)
def test_roundtrip_encode_decode(det_str, det_int):
    _ = det_int
    assert decode_detectorid(encode_detectorid(det_str)) == det_str


@pytest.mark.parametrize(("det_str", "det_int"), SPEC_CASES)
def test_roundtrip_decode_encode(det_str, det_int):
    _ = det_str
    assert encode_detectorid(decode_detectorid(det_int)) == det_int


# ---------------------------------------------------------------------------
# Legacy DUMMY single-digit parsing (encode only - decode always uses 2 digits)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("legacy", "canonical_int"),
    [
        ("DUMMY0", 0x0D000000),
        ("DUMMY9", 0x0D000090),
    ],
)
def test_encode_dummy_legacy_single_digit(legacy, canonical_int):
    assert encode_detectorid(legacy) == canonical_int


def test_decode_dummy_always_two_digits():
    # Integer that maps to DUMMY00 should decode to "DUMMY00", never "DUMMY0"
    assert decode_detectorid(0x0D000000) == "DUMMY00"
    assert decode_detectorid(0x0D000090) == "DUMMY09"


# ---------------------------------------------------------------------------
# HPGe sub-serial letter boundary values
# ---------------------------------------------------------------------------


def test_hpge_sub_serial_a():
    # A = 0
    assert encode_detectorid("C00000A") == 0x01000000
    assert decode_detectorid(0x01000000) == "C00000A"


def test_hpge_sub_serial_p():
    # P = 15 = 0xF
    assert encode_detectorid("V99999P") == (0x04 << 24) | (99999 << 4) | 0xF
    assert decode_detectorid((0x04 << 24) | (99999 << 4) | 0xF) == "V99999P"


# ---------------------------------------------------------------------------
# Special C variants: all supported digits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(10))
def test_encode_c00ang(n):
    det_str = f"C00ANG{n}"
    expected = (0x01 << 24) | ((0xF1000 | n) << 4)
    assert encode_detectorid(det_str) == expected


@pytest.mark.parametrize("n", range(10))
def test_decode_c00ang(n):
    value = (0x01 << 24) | ((0xF1000 | n) << 4)
    assert decode_detectorid(value) == f"C00ANG{n}"


@pytest.mark.parametrize("n", range(10))
def test_encode_c000rg(n):
    det_str = f"C000RG{n}"
    expected = (0x01 << 24) | ((0xF2000 | n) << 4)
    assert encode_detectorid(det_str) == expected


@pytest.mark.parametrize("n", range(10))
def test_decode_c000rg(n):
    value = (0x01 << 24) | ((0xF2000 | n) << 4)
    assert decode_detectorid(value) == f"C000RG{n}"


# ---------------------------------------------------------------------------
# encode_detectorid input validation
# ---------------------------------------------------------------------------


def test_encode_type_error():
    with pytest.raises(TypeError):
        encode_detectorid(42)


def test_encode_empty_string():
    with pytest.raises(ValueError):
        encode_detectorid("")


def test_encode_invalid_string():
    with pytest.raises(ValueError):
        encode_detectorid("INVALID")


def test_encode_hpge_wrong_digit_count():
    # 4 digits instead of 5
    with pytest.raises(ValueError):
        encode_detectorid("C0000A")


def test_encode_hpge_bad_sub_letter():
    # Q is beyond P (nibble max = 15 = P)
    with pytest.raises(ValueError):
        encode_detectorid("C00000Q")


def test_encode_sipm_wrong_digit_count():
    with pytest.raises(ValueError):
        encode_detectorid("S99")  # 2 digits instead of 3


def test_encode_pmt_wrong_digit_count():
    with pytest.raises(ValueError):
        encode_detectorid("PMT99")  # 2 digits instead of 3


def test_encode_puls_wrong_digit_count():
    with pytest.raises(ValueError):
        encode_detectorid("PULS9")  # 1 digit instead of 2


def test_encode_aux_wrong_digit_count():
    with pytest.raises(ValueError):
        encode_detectorid("AUX9")


def test_encode_bsln_wrong_digit_count():
    with pytest.raises(ValueError):
        encode_detectorid("BSLN9")


def test_encode_muon_wrong_digit_count():
    with pytest.raises(ValueError):
        encode_detectorid("MUON9")


def test_encode_dummy_three_digits_rejected():
    # Three digits is not part of the spec (only 1 legacy or 2 standard)
    with pytest.raises(ValueError):
        encode_detectorid("DUMMY100")


def test_encode_c00ang_non_digit_suffix():
    with pytest.raises(ValueError):
        encode_detectorid("C00ANGA")  # must end with a digit 0-9


def test_encode_c000rg_non_digit_suffix():
    with pytest.raises(ValueError):
        encode_detectorid("C000RGA")


# ---------------------------------------------------------------------------
# decode_detectorid input validation
# ---------------------------------------------------------------------------


def test_decode_type_error():
    with pytest.raises(TypeError):
        decode_detectorid("0x01000000")


def test_decode_float_rejected():
    with pytest.raises(TypeError):
        decode_detectorid(1.0)


def test_decode_negative():
    with pytest.raises(ValueError):
        decode_detectorid(-1)


def test_decode_out_of_32bit_range():
    with pytest.raises(ValueError):
        decode_detectorid(0x100000000)


def test_decode_reserved_r_nibble():
    with pytest.raises(ValueError):
        decode_detectorid(0x10000000)  # R nibble = 1


def test_decode_reserved_type_zero():
    with pytest.raises(ValueError):
        decode_detectorid(0x00000000)  # T = 0 (reserved)


def test_decode_reserved_type_5():
    with pytest.raises(ValueError):
        decode_detectorid(0x05000000)


def test_decode_reserved_type_6():
    with pytest.raises(ValueError):
        decode_detectorid(0x06000000)


def test_decode_reserved_type_7():
    with pytest.raises(ValueError):
        decode_detectorid(0x07000000)


def test_decode_reserved_type_8():
    with pytest.raises(ValueError):
        decode_detectorid(0x08000000)


def test_decode_hpge_serial_out_of_range():
    # Serial > 99999 for normal HPGe (not a special C case)
    with pytest.raises(ValueError):
        decode_detectorid((0x01 << 24) | (100000 << 4))


def test_decode_sipm_serial_out_of_range():
    with pytest.raises(ValueError):
        decode_detectorid(0x09000000 | (1000 << 4))


def test_decode_pmt_serial_out_of_range():
    with pytest.raises(ValueError):
        decode_detectorid(0x0A000000 | (1000 << 4))


def test_decode_puls_invalid_sub_serial():
    with pytest.raises(ValueError):
        decode_detectorid(0x0B000002)  # y = 2, only 0 or 1 valid


def test_decode_puls_serial_out_of_range():
    with pytest.raises(ValueError):
        decode_detectorid(0x0B000000 | (100 << 4))


def test_decode_aux_serial_out_of_range():
    with pytest.raises(ValueError):
        decode_detectorid(0x0C000000 | (100 << 4))


def test_decode_aux_nonzero_sub_serial():
    with pytest.raises(ValueError):
        decode_detectorid(0x0C000001)


def test_decode_dummy_serial_out_of_range():
    with pytest.raises(ValueError):
        decode_detectorid(0x0D000000 | (100 << 4))


def test_decode_bsln_serial_out_of_range():
    with pytest.raises(ValueError):
        decode_detectorid(0x0E000000 | (100 << 4))


def test_decode_muon_serial_out_of_range():
    with pytest.raises(ValueError):
        decode_detectorid(0x0F000000 | (100 << 4))


def test_decode_c00ang_nonzero_sub_serial():
    # C00ANG0 with Y=1 -> invalid
    with pytest.raises(ValueError):
        decode_detectorid((0x01 << 24) | (0xF1000 << 4) | 1)


def test_decode_c000rg_nonzero_sub_serial():
    with pytest.raises(ValueError):
        decode_detectorid((0x01 << 24) | (0xF2000 << 4) | 1)


def test_decode_c00ang_digit_out_of_range():
    # Serial upper nibbles = 0xF100, last nibble = 0xA (>9)
    with pytest.raises(ValueError):
        decode_detectorid((0x01 << 24) | ((0xF1000 | 0xA) << 4))


def test_decode_c000rg_digit_out_of_range():
    with pytest.raises(ValueError):
        decode_detectorid((0x01 << 24) | ((0xF2000 | 0xA) << 4))


# ---------------------------------------------------------------------------
# numpy integer acceptance
# ---------------------------------------------------------------------------


def test_decode_numpy_int():
    np = pytest.importorskip("numpy")
    assert decode_detectorid(np.uint32(0x02000002)) == "B00000C"
    assert decode_detectorid(np.int64(0x09002780)) == "S632"
