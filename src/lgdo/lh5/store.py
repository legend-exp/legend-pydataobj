"""
This module implements routines from reading and writing LEGEND Data Objects in
HDF5 files.
"""

from __future__ import annotations

import bisect
import logging
import numpy as np
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import h5py
from numpy.typing import ArrayLike

from .. import types
from . import _serializers, utils

log = logging.getLogger(__name__)


class LH5Store:
    """
    Class to represent a store of LEGEND HDF5 files. The two main methods
    implemented by the class are :meth:`read` and :meth:`write`.

    Examples
    --------
    >>> from lgdo import LH5Store
    >>> store = LH5Store()
    >>> obj, _ = store.read("/geds/waveform", "file.lh5")
    >>> type(obj)
    lgdo.waveformtable.WaveformTable
    """

    def __init__(
        self, 
        base_path: str = "", 
        keep_open: bool = False, 
        metacachesize: int = 100,
    ) -> None:
        """
        Parameters
        ----------
        base_path
            directory path to prepend to LH5 files.
        keep_open
            whether to keep files open by storing the :mod:`h5py` objects as
            class attributes.
        metacache
            maximum size for metadata cache, default is 100 MB (specify in units of MB)
        """
        self.base_path = "" if base_path == "" else utils.expand_path(base_path)
        self.keep_open = keep_open
        self.files = {}
        self.metadata_cache = {}
        self.metacachesize = int(metacachesize * 1E6)

    def gimme_file(self, lh5_file: str | h5py.File, mode: str = "r") -> h5py.File:
        """Returns a :mod:`h5py` file object from the store or creates a new one.

        Parameters
        ----------
        lh5_file
            LH5 file name.
        mode
            mode in which to open file. See :class:`h5py.File` documentation.
        """
        if isinstance(lh5_file, h5py.File):
            return lh5_file

        if mode == "r":
            lh5_file = utils.expand_path(lh5_file, base_path=self.base_path)

        if lh5_file in self.files:
            return self.files[lh5_file]

        if self.base_path != "":
            full_path = os.path.join(self.base_path, lh5_file)
        else:
            full_path = lh5_file

        if mode != "r":
            directory = os.path.dirname(full_path)
            if directory != "" and not os.path.exists(directory):
                log.debug(f"making path {directory}")
                os.makedirs(directory)

        if mode == "r" and not os.path.exists(full_path):
            msg = f"file {full_path} not found"
            raise FileNotFoundError(msg)

        if mode != "r" and os.path.exists(full_path):
            log.debug(f"opening existing file {full_path} in mode '{mode}'")

        h5f = h5py.File(full_path, mode)

        if self.keep_open:
            self.files[lh5_file] = h5f

        return h5f

    def gimme_group(
        self,
        group: str | h5py.Group,
        base_group: h5py.Group,
        grp_attrs: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> h5py.Group:
        """
        Returns an existing :class:`h5py` group from a base group or creates a new one.

        See Also
        --------
        .lh5.utils.get_h5_group
        """
        return utils.get_h5_group(group, base_group, grp_attrs, overwrite)

    def get_buffer(
        self,
        name: str,
        lh5_file: str | h5py.File | Sequence[str | h5py.File],
        size: int | None = None,
        field_mask: Mapping[str, bool] | Sequence[str] | None = None,
    ) -> types.LGDO:
        """Returns an LH5 object appropriate for use as a pre-allocated buffer
        in a read loop. Sets size to `size` if object has a size.
        """
        obj, n_rows = self.read(name, lh5_file, n_rows=0, field_mask=field_mask)
        if hasattr(obj, "resize") and size is not None:
            obj.resize(new_size=size)
        return obj

    def read(
        self,
        name: str,
        lh5_file: str | h5py.File | Sequence[str | h5py.File],
        start_row: int = 0,
        n_rows: int = sys.maxsize,
        idx: ArrayLike = None,
        use_h5idx: bool = False,
        field_mask: Mapping[str, bool] | Sequence[str] | None = None,
        obj_buf: types.LGDO = None,
        obj_buf_start: int = 0,
        decompress: bool = True,
    ) -> tuple[types.LGDO, int]:
        """Read LH5 object data from a file in the store.

        See Also
        --------
        .lh5.core.read
        """
        metadata = None

        # grab files from store
        if not isinstance(lh5_file, (str, h5py.File)):
            lh5_file = [self.gimme_file(f, "r") for f in list(lh5_file)]
        else:
            lh5_file = self.gimme_file(lh5_file, "r")
            metadata = self.load_metadata(lh5_file, name) 

        h5f = lh5_file
        # Handle list-of-files recursively - how about no?
        if not isinstance(h5f, (str, h5py.File)):
            thislh5_file = list(h5f)
            n_rows_read = 0

            for i, h5f in enumerate(thislh5_file):
                metadata = self.load_metadata(h5f, name) 

                if isinstance(idx, list) and len(idx) > 0 and not np.isscalar(idx[0]):
                    # a list of lists: must be one per file
                    idx_i = idx[i]
                elif idx is not None:
                    # make idx a proper tuple if it's not one already
                    if not (isinstance(idx, tuple) and len(idx) == 1):
                        idx = (idx,)
                    # idx is a long continuous array
                    n_rows_i = utils.read_n_rows(name, h5f, metadata=metadata)
                    # find the length of the subset of idx that contains indices
                    # that are less than n_rows_i
                    n_rows_to_read_i = bisect.bisect_left(idx[0], n_rows_i)
                    # now split idx into idx_i and the remainder
                    idx_i = (idx[0][:n_rows_to_read_i],)
                    idx = (idx[0][n_rows_to_read_i:] - n_rows_i,)
                else:
                    idx_i = None
                n_rows_i = n_rows - n_rows_read                

                obj_buf, n_rows_read_i = _serializers._h5_read_lgdo(
                    name,
                    h5f,
                    start_row=start_row,
                    n_rows=n_rows_i,
                    idx=idx_i,
                    use_h5idx=use_h5idx,
                    field_mask=field_mask,
                    obj_buf=obj_buf,
                    obj_buf_start=obj_buf_start,
                    decompress=decompress,
                    metadata=metadata,
                )

                n_rows_read += n_rows_read_i
                if n_rows_read >= n_rows or obj_buf is None:
                    return obj_buf, n_rows_read
                start_row = 0
                obj_buf_start += n_rows_read_i
            
            return obj_buf, n_rows_read
                
        return _serializers._h5_read_lgdo(
            name,
            lh5_file,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            use_h5idx=use_h5idx,
            field_mask=field_mask,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
            decompress=decompress,
            metadata=metadata,
        )

    def write(
        self,
        obj: types.LGDO,
        name: str,
        lh5_file: str | h5py.File,
        group: str | h5py.Group = "/",
        start_row: int = 0,
        n_rows: int | None = None,
        wo_mode: str = "append",
        write_start: int = 0,
        **h5py_kwargs,
    ) -> None:
        """Write an LGDO into an LH5 file.

        See Also
        --------
        .lh5.core.write
        """
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
            raise ValueError(msg)

        # "mode" is for the h5df.File and wo_mode is for this function
        # In hdf5, 'a' is really "modify" -- in addition to appending, you can
        # change any object in the file. So we use file:append for
        # write_object:overwrite.
        mode = "w" if wo_mode == "of" else "a"

        return _serializers._h5_write_lgdo(
            obj,
            name,
            self.gimme_file(lh5_file, mode=mode),
            group=group,
            start_row=start_row,
            n_rows=n_rows,
            wo_mode=wo_mode,
            write_start=write_start,
            **h5py_kwargs,
        )

    def write_metadata(
        self,
        lh5_file: str | h5py.File | list,
    ) -> dict:
        """Writes the `"metadata"` dataset to an LH5 file. Gathers the metadata from the file and writes it.
        Overwrites any existing `"metadata"` dataset. Takes an LH5 file, path to file, or list of either.

        Call this method after you are finished writing to a file in order to make reading from the file faster.
        It takes some time to build the metadata so it is best to call this when you are done writing.
        """

        # grab files from store
        if not isinstance(lh5_file, (str, h5py.File)):
            lh5_files = [self.gimme_file(f, "a") for f in list(lh5_file)]
        else:
            lh5_files = [self.gimme_file(lh5_file, "a")]

        for lh5_file in lh5_files:
            # delete the old metadata if it exists
            if "metadata" in lh5_file:
                del lh5_file["metadata"]
                log.debug(
                    f"deleted old metadata from {lh5_file.filename}"
                )    

            # get the new metadata from the file
            metadata = utils.get_metadata(lh5_file=lh5_file, force=True)

            # write the metadata as a JSON string
            jsontowrite = str(metadata).replace("'", "\"")
            lh5_file.create_dataset(f'metadata', dtype=f'S{len(str(jsontowrite))}', data=str(jsontowrite))
            lh5_file['metadata'].attrs['datatype'] = 'JSON'
            log.debug(
                f"wrote metadata to {lh5_file.filename}"
            )   

        return metadata

    def read_n_rows(
        self, 
        name: str, 
        lh5_file: str | h5py.File,
        metadata: dict | None = None,
    ) -> int | None:
        """Look up the number of rows in an Array-like object called `name` in `lh5_file`.

        Return ``None`` if it is a :class:`.Scalar` or a :class:`.Struct`.
        """
        # check if metadata exists
        if metadata is None:
            metadata = self.load_metadata(lh5_file, name)
        return utils.read_n_rows(name, self.gimme_file(lh5_file, "r"), metadata=metadata)

    def load_metadata(
        self,
        lh5_file: str | h5py.File,
        name: str,
    )-> dict:
        """Gets metadata dataset from the metadata cache in the store if it exists or else from the file. 
        Returns `None` if metadata is not found in the file or if the request `name` is missing from the metadata."""
        metadata = None

        lh5_file = self.gimme_file(lh5_file, "r")

        incache = False
        if lh5_file.filename in self.metadata_cache:
            # if the file was modified since we read in the metadata, read in new metadata
            if os.path.getmtime(lh5_file.filename) <= self.metadata_cache[lh5_file.filename]["timestamp"]:
                metadata = self.metadata_cache[lh5_file.filename]["metadata"]
                incache = True
                log.debug(
                    f"read metadata from cache for {lh5_file.filename}"
                )
            else:
                log.debug(
                    f"cached metadata for {lh5_file.filename} is old, getting new metadata"
                )               

        if not incache:
            log.debug(
                f"metadata for {lh5_file.filename} not in cache"
            )
            # don't want to build the metadata because this is slow if you only want to grab a few datasets
            metadata = utils.get_metadata(lh5_file, build=False)
            
            if metadata is not None:
                # it is possible that someone could read a file, then want to write to it, then read it again using the
                # same store instance. If we keep the metadata from before the file were updated, we would not find some 
                # newly written information. We will check if the file has been modified since the last time we read
                # in the metadata.
                self.metadata_cache[lh5_file.filename] = {
                    "metadata":metadata, 
                    "timestamp":os.path.getmtime(lh5_file.filename)}
                log.debug(
                    f"added metadata to cache for {lh5_file.filename}"
                )
                # clear the cache if we store something to the metadata cache so it is not too big
                self.clear_metadata_cache()

        if metadata is not None:
            # in case the metadata is broken or missing something
            try:
                metadata = utils.getFromDict(metadata, list(filter(None, name.strip('/').split('/'))))
                log.debug(
                    f"found {name} in metadata for {lh5_file.filename}"
                )
            except KeyError as e:
                metadata = None
                log.debug(
                    f"{e}"
                    f"did not find {name} in metadata for {lh5_file.filename}, setting metadata to None"
                )                    
        return metadata

    def clear_metadata_cache(
        self,
        forceclear: bool = False,
    ) -> None:
        """Removes entries from the metadata cache if the size is too large. Keeps at least one file in the cache
        regardless of its size. Default maximum size is 100 MB (specify `maxsize` in MB)."""
        if forceclear:
            files = list(self.metadata_cache.keys())
            for file in files:
                del self.metadata_cache[file]    

        elif len(self.metadata_cache) > 1 and (
            metadatasize := utils.getsize(self.metadata_cache)) > self.metacachesize:
            log.debug(
                f"metadata cache is {utils.fmtbytes(metadatasize)}, larger than max size of "
                f"{utils.fmtbytes(self.metacachesize)}, deleting entries to reduce size"
            )
            files = list(self.metadata_cache.keys())
            # in order of how the files were added to the metadata (so presumably the access order)
            while len(self.metadata_cache) > 0 and (utils.getsize(self.metadata_cache) > self.metacachesize):
                for file in files:
                    del self.metadata_cache[file]

        return
            