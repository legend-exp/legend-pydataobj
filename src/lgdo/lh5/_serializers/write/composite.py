from __future__ import annotations

import logging

import h5py

from .... import compression, types
from ... import datatype, utils
from ...exceptions import LH5EncodeError
from .array import _h5_write_array
from .scalar import _h5_write_scalar
from .vector_of_vectors import _h5_write_vector_of_vectors

log = logging.getLogger(__name__)


def _h5_write_lgdo(
    obj,
    name,
    lh5_file,
    group="/",
    start_row=0,
    n_rows=None,
    wo_mode="append",
    write_start=0,
    **h5py_kwargs,
):
    assert isinstance(obj, types.LGDO)

    if wo_mode == "write_safe":
        wo_mode = "w"
    if wo_mode == "append":
        wo_mode = "a"
    if wo_mode == "overwrite":
        wo_mode = "o"
    if wo_mode == "overwrite_file":
        wo_mode = "of"
        write_start = 0
    if wo_mode == "append_column":
        wo_mode = "ac"
    if wo_mode not in ["w", "a", "o", "of", "ac"]:
        msg = f"unknown wo_mode '{wo_mode}'"
        raise LH5EncodeError(msg, lh5_file, group, name)

    # "mode" is for the h5df.File and wo_mode is for this function
    # In hdf5, 'a' is really "modify" -- in addition to appending, you can
    # change any object in the file. So we use file:append for
    # write_object:overwrite.
    mode = "w" if wo_mode == "of" else "a"

    if not isinstance(lh5_file, h5py.File):
        lh5_file = h5py.File(lh5_file, mode=mode)

    log.debug(
        f"writing {obj!r}[{start_row}:{n_rows}] as "
        f"{lh5_file.filename}:{group}/{name}[{write_start}:], "
        f"mode = {wo_mode}, h5py_kwargs = {h5py_kwargs}"
    )

    group = utils.get_h5_group(group, lh5_file)

    if wo_mode == "w" and name in group:
        msg = f"can't overwrite '{name}' in wo_mode 'write_safe'"
        raise LH5EncodeError(msg, lh5_file, group, name)

    # struct or table or waveform table
    if isinstance(obj, types.Struct):
        return _h5_write_struct(
            obj,
            name,
            lh5_file,
            group=group,
            start_row=start_row,
            n_rows=n_rows,
            wo_mode=wo_mode,
            write_start=write_start,
            **h5py_kwargs,
        )

    # scalars
    if isinstance(obj, types.Scalar):
        return _h5_write_scalar(obj, name, lh5_file, group, wo_mode)

    # vector of encoded vectors
    if isinstance(
        obj, (types.VectorOfEncodedVectors, types.ArrayOfEncodedEqualSizedArrays)
    ):
        group = utils.get_h5_group(
            name, group, grp_attrs=obj.attrs, overwrite=(wo_mode == "o")
        )

        # ask not to further compress flattened_data, it is already compressed!
        obj.encoded_data.flattened_data.attrs["compression"] = None

        _h5_write_vector_of_vectors(
            obj.encoded_data,
            "encoded_data",
            lh5_file,
            group=group,
            start_row=start_row,
            n_rows=n_rows,
            wo_mode=wo_mode,
            write_start=write_start,
            **h5py_kwargs,
        )

        if isinstance(obj.decoded_size, types.Scalar):
            _h5_write_scalar(
                obj.decoded_size,
                "decoded_size",
                lh5_file,
                group=group,
                wo_mode=wo_mode,
            )
        else:
            _h5_write_array(
                obj.decoded_size,
                "decoded_size",
                lh5_file,
                group=group,
                start_row=start_row,
                n_rows=n_rows,
                wo_mode=wo_mode,
                write_start=write_start,
                **h5py_kwargs,
            )

        return None

    # vector of vectors
    if isinstance(obj, types.VectorOfVectors):
        return _h5_write_vector_of_vectors(
            obj,
            name,
            lh5_file,
            group=group,
            start_row=start_row,
            n_rows=n_rows,
            wo_mode=wo_mode,
            write_start=write_start,
            **h5py_kwargs,
        )

    # if we get this far, must be one of the Array types
    if isinstance(obj, types.Array):
        return _h5_write_array(
            obj,
            name,
            lh5_file,
            group=group,
            start_row=start_row,
            n_rows=n_rows,
            wo_mode=wo_mode,
            write_start=write_start,
            **h5py_kwargs,
        )

    msg = f"do not know how to write '{name}' of type '{type(obj).__name__}'"
    raise LH5EncodeError(msg, lh5_file, group, name)


def _h5_write_struct(
    obj,
    name,
    lh5_file,
    group="/",
    start_row=0,
    n_rows=None,
    wo_mode="append",
    write_start=0,
    **h5py_kwargs,
):
    assert isinstance(obj, types.Struct)

    # In order to append a column, we need to update the
    # `table{old_fields}` value in `group.attrs['datatype"]` to include
    # the new fields.  One way to do this is to override
    # `obj.attrs["datatype"]` to include old and new fields. Then we
    # can write the fields to the table as normal.
    if wo_mode == "ac":
        old_group = utils.get_h5_group(name, group)
        lgdotype = datatype.datatype(old_group.attrs["datatype"])
        fields = datatype.get_struct_fields(old_group.attrs["datatype"])
        if not issubclass(lgdotype, types.Struct):
            msg = f"Trying to append columns to an object of type {lgdotype.__name__}"
            raise LH5EncodeError(msg, lh5_file, group, name)

        # If the mode is `append_column`, make sure we aren't appending
        # a table that has a column of the same name as in the existing
        # table. Also make sure that the field we are adding has the
        # same size
        if len(list(set(fields).intersection(set(obj.keys())))) != 0:
            msg = (
                f"Can't append {list(set(fields).intersection(set(obj.keys())))} "
                "column(s) to a table with the same field(s)"
            )
            raise LH5EncodeError(msg, lh5_file, group, name)
        # It doesn't matter what key we access, as all fields in the old table have the same size
        if old_group[next(iter(old_group.keys()))].size != obj.size:
            msg = (
                f"Table sizes don't match. Trying to append column of size {obj.size} "
                f"to a table of size {old_group[next(iter(old_group.keys()))].size}."
            )
            raise LH5EncodeError(msg, lh5_file, group, name)

        # Now we can append the obj.keys() to the old fields, and then update obj.attrs.
        fields.extend(list(obj.keys()))
        obj.attrs.pop("datatype")
        obj.attrs["datatype"] = "table" + "{" + ",".join(fields) + "}"

    group = utils.get_h5_group(
        name,
        group,
        grp_attrs=obj.attrs,
        overwrite=(wo_mode in ["o", "ac"]),
    )
    # If the mode is overwrite, then we need to peek into the file's
    # table's existing fields.  If we are writing a new table to the
    # group that does not contain an old field, we should delete that
    # old field from the file
    if wo_mode == "o":
        # Find the old keys in the group that are not present in the
        # new table's keys, then delete them
        for key in list(set(group.keys()) - set(obj.keys())):
            log.debug(f"{key} is not present in new table, deleting field")
            del group[key]

    for field in obj:
        # eventually compress waveform table values with LGDO's
        # custom codecs before writing
        # if waveformtable.values.attrs["compression"] is NOT a
        # WaveformCodec, just leave it there
        obj_fld = None
        if (
            isinstance(obj, types.WaveformTable)
            and field == "values"
            and not isinstance(obj.values, types.VectorOfEncodedVectors)
            and not isinstance(obj.values, types.ArrayOfEncodedEqualSizedArrays)
            and "compression" in obj.values.attrs
            and isinstance(obj.values.attrs["compression"], compression.WaveformCodec)
        ):
            codec = obj.values.attrs["compression"]
            obj_fld = compression.encode(obj.values, codec=codec)
        else:
            obj_fld = obj[field]

        # Convert keys to string for dataset names
        f = str(field)
        _h5_write_lgdo(
            obj_fld,
            f,
            lh5_file,
            group=group,
            start_row=start_row,
            n_rows=n_rows,
            wo_mode=wo_mode,
            write_start=write_start,
            **h5py_kwargs,
        )
