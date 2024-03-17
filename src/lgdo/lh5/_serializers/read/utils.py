from __future__ import annotations

from ...exceptions import LH5DecodeError


def check_obj_buf_attrs(attrs, new_attrs, file, name):
    if set(attrs.keys()) != set(new_attrs.keys()):
        msg = (
            f"existing buffer and new data chunk have different attributes: "
            f"obj_buf.attrs={attrs} != {file.filename}[{name}].attrs={new_attrs}"
        )
        raise LH5DecodeError(msg, file, name)
