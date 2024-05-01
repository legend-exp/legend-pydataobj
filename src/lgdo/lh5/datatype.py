from __future__ import annotations

import re
from collections import OrderedDict

from .. import types as lgdo

_lgdo_datatype_map: dict[str, lgdo.LGDO] = OrderedDict(
    [
        (lgdo.Scalar, r"^real$|^bool$|^complex$|^bool$|^string$"),
        (lgdo.VectorOfVectors, r"^array<1>\{array<1>\{.+\}\}$"),
        (lgdo.VectorOfEncodedVectors, r"^array<1>\{encoded_array<1>\{.+\}\}$"),
        (
            lgdo.ArrayOfEncodedEqualSizedArrays,
            r"^array_of_encoded_equalsized_arrays<1,1>\{.+\}$",
        ),
        (lgdo.Struct, r"^struct\{.*\}$"),
        (lgdo.Table, r"^table\{.*\}$"),
        (lgdo.FixedSizeArray, r"^fixedsize_array<\d+>\{.+\}$"),
        (lgdo.ArrayOfEqualSizedArrays, r"^array_of_equalsized_arrays<1,1>\{.+\}$"),
        (lgdo.Array, r"^array<\d+>\{.+\}$"),
    ]
)
"""Mapping between LGDO types and regular expression defining the corresponding datatype string"""


def datatype(expr: str) -> type:
    """Return the LGDO type corresponding to a datatype string."""
    expr = expr.strip()
    for type_, regex in _lgdo_datatype_map.items():
        if re.search(regex, expr):
            return type_

    msg = f"unknown datatype '{expr}'"
    raise RuntimeError(msg)


def get_nested_datatype_string(expr: str) -> str:
    """Matches the content of the outermost curly brackets."""
    return re.search(r"\{(.*)\}$", expr).group(1)


def get_struct_fields(expr: str) -> list[str]:
    """Returns a list of :class:`~.lgdo.types.struct.Struct` fields, given its datatype string."""
    fields = get_nested_datatype_string(expr).split(",")
    return [] if fields == [""] else fields
