from __future__ import annotations


def check_obj_buf_attrs(attrs, new_attrs, name):
    if set(attrs.keys()) != set(new_attrs.keys()):
        msg = (
            f"existing LGDO buffer and new data chunk have different attributes: "
            f"obj_buf.attrs={attrs} != {name}.attrs={new_attrs}"
        )
        raise RuntimeError(msg)
