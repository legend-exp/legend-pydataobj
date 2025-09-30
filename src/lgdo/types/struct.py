"""
Implements a LEGEND Data Object representing a struct and corresponding
utilities.
"""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from itertools import chain
from typing import Any

import numpy as np

from .lgdo import LGDO

log = logging.getLogger(__name__)
# parse for name1/name2 or name1.name2
parser = re.compile("([^\\./]*)(?:[\\./](.+))?")


class Struct(LGDO, MutableMapping):
    """A dictionary of LGDO's with an optional set of attributes.

    After instantiation, add fields using :meth:`add_field` to keep the
    datatype updated, or call :meth:`update_datatype` after adding.
    """

    def __new__(cls, *args, **kwargs) -> Struct:
        obj = super().__new__(cls, *args, **kwargs)
        obj.obj_dict = {}
        return obj

    def __init__(
        self,
        obj_dict: Mapping[str, LGDO] | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        obj_dict
            instantiate this Struct using the supplied named LGDO's.  Note: no
            copy is performed, the objects are used directly.
        attrs
            a set of user attributes to be carried along with this LGDO.
        """
        if obj_dict is not None:
            self.update(obj_dict)

        # check the datatype attribute passed by the user and sort the fields
        # to ensure consistent behavior
        if attrs is not None and "datatype" in attrs:
            _attrs = copy.copy(dict(attrs))

            if not _is_struct_datatype(self.datatype_name(), _attrs["datatype"]):
                msg = (
                    f"datatype attribute ({self.attrs['datatype']}) is not "
                    f"compatible with class datatype!"
                )
                raise ValueError(msg)

            _attrs["datatype"] = _sort_datatype_fields(_attrs["datatype"])
            attrs = _attrs

        super().__init__(attrs)

    def datatype_name(self) -> str:
        return "struct"

    def form_datatype(self) -> str:
        return (
            self.datatype_name()
            + "{"
            + ",".join(sorted([str(k) for k in self.keys()]))
            + "}"
        )

    def update_datatype(self) -> None:
        self.attrs["datatype"] = self.form_datatype()

    def add_field(self, name: str | int, obj: LGDO | Mapping[str, LGDO]) -> None:
        """Add a field to the table or set an existing field.

        Parameters
        ----------
        name
            key to use for field. Key can be nested (e.g. ``name1.name2`` or
            ``name1/name2``); this will navigate through the tree, creating
            new fields as needed
        obj
            object to add. Can be any LGDO object, or a mapping from names
            to LGDO objects that will be converted to an LGDO :class:`.Struct`
        """
        name1, name2 = parser.match(name).groups()
        if name2:
            if not name1:
                self.add_field(name2, obj)
            else:
                if name1 not in self:
                    self.add_field(name1, Struct())
                self[name1].add_field(name2, obj)
        else:
            if not isinstance(obj, LGDO):
                if isinstance(obj, Mapping):
                    obj = Struct(obj)
                else:
                    msg = f"value of '{name}' ({obj!r}) is not an LGDO or a Mapping"
                    raise ValueError(msg)

            self.obj_dict[name1] = obj
            self.update_datatype()

    def __getitem__(self, name: str) -> LGDO:
        """Get value associated with field. Name can be nested (e.g. ``name1.name2``
        or ``name1/name2``); this will search in nested Structs
        """
        name1, name2 = parser.match(name).groups()
        obj = self.obj_dict[name1] if name1 else self
        return obj if not name2 else obj[name2]

    def __setitem__(self, name: str, obj: LGDO) -> None:
        return self.add_field(name, obj)

    def __delitem__(self, name: str) -> None:
        self.remove_field(name, delete=True)

    def __iter__(self) -> Iterator[str]:
        "Iterator over top level fields"
        return iter(self.obj_dict)

    def __len__(self) -> int:
        "Return number of fields"
        return len(self.obj_dict)

    def update(
        self, other: Struct | Mapping[str, LGDO] | Iterable[str, LGDO] = (), /, **kwargs
    ) -> None:
        """Add or set a field(s) to the table or set an existing field. For
        nested Structs, only update at the lowest level of nesting; unlike for
        nested dicts, nested fields not included in other will not be removed.

        Parameters
        ----------
        other
            Struct/Mapping from fields to new values
        """
        for k, v in chain(
            other.items() if isinstance(other, Mapping) else other, kwargs.items()
        ):
            if isinstance(v, Mapping) and k in self:
                self[k].update(v)
            else:
                self[k] = v

    # Note: in principle these are automatically defined by Mapping; however, they will get the length wrong for Tables
    def keys(self):
        return self.obj_dict.keys()

    def items(self):
        return self.obj_dict.items()

    def values(self):
        return self.obj_dict.values()

    def __getattr__(self, name: str) -> LGDO:
        if name in self.obj_dict:
            return self.obj_dict[name]

        raise AttributeError(name)

    def remove_field(self, name: str | int, delete: bool = False) -> None:
        """Remove a field from the table.

        Parameters
        ----------
        name
            name of the field to be removed.
        delete
            if ``True``, delete the field object by calling :any:`del`.
        """
        name1, name2 = parser.match(name).groups()
        if name2:
            if not name1:
                self.remove_field(name2, delete)
            else:
                self[name1].remove_field(name2, delete)

        else:
            if delete:
                del self.obj_dict[name1]
            else:
                self.pop(name1)
            self.update_datatype()

    def __str__(self) -> str:
        """Convert to string (e.g. for printing)."""

        thr_orig = np.get_printoptions()["threshold"]
        np.set_printoptions(threshold=8)

        string = "{\n"
        for k, v in self.items():
            if "\n" in str(v):
                rv = str(v).replace("\n", "\n    ")
                string += f" '{k}':\n    {rv},\n"
            else:
                string += f" '{k}': {v},\n"
        string += "}"

        attrs = self.getattrs()
        if attrs:
            string += f" with attrs={attrs}"

        np.set_printoptions(threshold=thr_orig)

        return string

    def __repr__(self) -> str:
        npopt = np.get_printoptions()
        np.set_printoptions(threshold=5, edgeitems=2, linewidth=100)
        out = (
            self.__class__.__name__
            + "(dict="
            + self.obj_dict.__repr__()
            + f", attrs={self.attrs!r})"
        )
        np.set_printoptions(**npopt)
        return " ".join(out.replace("\n", " ").split())

    def view_as(self) -> None:
        r"""View the Struct data as a third-party format data structure.

        Error
        -----
        Not implemented. Since Struct's fields can have different lengths,
        converting to a NumPy, Pandas or Awkward is generally not possible.
        Call :meth:`.LGDO.view_as` on the fields instead.

        See Also
        --------
        .LGDO.view_as
        """
        msg = (
            "Since Struct's fields can have different lengths, "
            "converting to a NumPy, Pandas or Awkward is generally "
            "not possible. Call view_as() on the fields instead."
        )
        raise NotImplementedError(msg)


def _is_struct_datatype(dt_name, expr):
    return re.search("^" + dt_name + r"\{(.*)\}$", expr) is not None


def _get_struct_fields(expr: str) -> list[str]:
    assert _is_struct_datatype(".*", expr)

    arr = re.search(r"\{(.*)\}$", expr).group(1).split(",")
    if arr == [""]:
        arr = []

    return sorted(arr)


def _struct_datatype_equal(dt_name, dt1, dt2):
    if any(not _is_struct_datatype(dt_name, dt) for dt in (dt1, dt2)):
        return False

    return _get_struct_fields(dt1) == _get_struct_fields(dt2)


def _sort_datatype_fields(expr):
    assert _is_struct_datatype(".*", expr)

    match = re.search(r"^(.*)\{.*\}$", expr)
    struct_type = match.group(1)
    fields = _get_struct_fields(expr)

    return struct_type + "{" + ",".join(sorted([str(k) for k in fields])) + "}"
