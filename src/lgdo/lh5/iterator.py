from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterator, Mapping
from concurrent.futures import Executor, ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import chain, product
from typing import Any

import awkward as ak
import numpy as np
import pandas as pd
from hist import Hist, axis
from numpy.typing import NDArray

from ..types import LGDOCollection, Table
from ..units import default_units_registry as ureg
from .store import LH5Store
from .utils import expand_path


class LH5Iterator(Iterator):
    """Iterate over chunks of entries from LH5 files.

    The iterator reads ``buffer_len`` entries at a time from one or more
    files.  The LGDO instance returned at each iteration is reused to avoid
    reallocations, so copy the data if it should be preserved.

    Examples
    --------
    Iterate through a table one chunk at a time and call ``process`` on each chunk::

            from lgdo.lh5 import LH5Iterator
            for table in LH5Iterator("data.lh5", "geds/raw/energy", buffer_len=100):
                process(table)

    ``LH5Iterator`` can also be used for random access::

            it = LH5Iterator(files, groups)
            table = it.read(i_entry)

    In case of multiple files or an entry selection, ``i_entry`` refers to the
    global event index across all files.

    When instantiating an iterator you must provide a list of files and the
    HDF5 groups to read.  Optional parameters allow field masking, event
    selection and pairing the iterator with a "friend" iterator that is read in
    parallel.  Several properties are available to obtain the provenance of the
    data currently loaded:

    - ``current_i_entry`` -- index within the entry list of the first entry in
      the buffer
    - ``current_local_entries`` -- entry numbers relative to the file the data
      came from
    - ``current_global_entries`` -- entry number relative to the full dataset
    - ``current_files`` -- file name corresponding to each entry in the buffer
    - ``current_groups`` -- group name corresponding to each entry in the
      buffer
    """

    def __init__(
        self,
        lh5_files: str | Collection[str | Collection[str]],
        groups: str | Collection[str | Collection[str]],
        *,
        base_path: str = "",
        entry_list: Collection[int] | Collection[Collection[int]] = None,
        entry_mask: Collection[bool] | Collection[Collection[bool]] = None,
        i_start: int = 0,
        n_entries: int = None,
        field_mask: Mapping[str, bool] | Collection[str] = None,
        group_data: Mapping[Collection] | ak.Array = None,
        buffer_len: int = "100*MB",
        file_cache: int = 10,
        ds_map: NDArray[int] = None,
        friend: Collection[LH5Iterator] = None,
        friend_prefix: str = "",
        friend_suffix: str = "",
        h5py_open_mode: str = "r",
    ) -> None:
        """
        Constructor for LH5Iterator. Must provide a file or collection of
        files, and an lh5 group or collection of groups to read data from.

        Collections of files and groups can be nested. At the top level, we
        expect the same number of entries (one set of files to one set of groups).
        For each corresponding pair of sets, we will loop over each pairing of
        a file and group, with an inner loop over the groups to minimize the
        opening of files. Wildcards used for files will be expanded and applied
        in the inner loop (i.e. each file in a wildcard will read the same groups).
        If groups is an un-nested collection of strings, use all groups for all files.

        Examples
        --------
        Read "ch1/table" and "ch2/table" from "file.lh5"::

            LH5Iterator("/path/to/file.lh5", ["ch1/table", "ch2/table"])

        Read "ch1" from all lh5 files in "/path1", then read "ch1" and "ch2" from
        "/path2/file.lh5", and then read "ch1/2/3" from both "file1.lh5" and "file2.lh5"::

            LH5Iterator(
                ["/path1/*.lh5", "/path2/file.lh5", ["/path3/file1.lh5", "/path3/file2.lh5"]],
                ["ch1/table", ["ch1/table", "ch2/table"], ["ch1/table, "ch2/table, "ch3/table"]]
            )

        Parameters
        ----------
        lh5_files
            file(s) to read from (see above). May include wildcards and environment variables.
        groups
            HDF5 group(s) to read (see above).
        entry_list
            list of entry numbers to read. If a nested list is provided,
            expect one top-level list for each file, containing a list of
            local entries. If a list of ints is provided, use global entries.
        entry_mask
            mask of entries to read. If a list of arrays is provided, expect
            one for each file. Ignore if a selection list is provided.
        i_start
            index of first entry to start at when iterating
        n_entries
            number of entries to read before terminating iteration
        field_mask
            mask of which fields to read. See :meth:`LH5Store.read` for
            more details.
        group_data
            mapping of values corresponding to each provided lh5 group. Values
            will be duplicated for each entry in each dataset, corresponding to
            the correct group, and added to the output table. This should have
            same structure as ``groups``.
        buffer_len
            number of entries in tables yielded by iterator. Can be provided as
            a value with a unit of memory; in this case, use the estimated number
            of rows that will yield tables that require the provided memory.
            Default to 100*MB.
        file_cache
            maximum number of files to keep open at a time
        ds_map
            cumulative entries in datasets corresponding to file/group pairs.
            This can be provided on construction to speed up random or sparse
            access; otherwise, we sequentially read the size of each group.
            WARNING: no checks for accuracy are performed so only use this if
            you know what you are doing!
        friend
            a \"friend\" LH5Iterator that will be joined to this one, and read
            in parallel. The friend should have the same length and entry list.
            Each iteration will return a single LH5 Table containing columns from
            both iterators. The buffer_len will be set to the minimum of the two.
        friend_prefix
            prefix for fields in friend iterator for resolving naming conflicts
        friend_suffix
            suffix for fields in friend iterator for resolving naming conflicts
        h5py_open_mode
            file open mode used when acquiring file handles. ``r`` (default)
            opens files read-only while ``a`` allow opening files for
            write-appending as well.
        """

        if h5py_open_mode == "read":
            h5py_open_mode = "r"
        if h5py_open_mode == "append":
            h5py_open_mode = "a"
        if h5py_open_mode not in ["r", "a"]:
            msg = f"unknown h5py_open_mode '{h5py_open_mode}'"
            raise ValueError(msg)

        self.lh5_st = LH5Store(
            base_path=base_path, keep_open=file_cache, default_mode=h5py_open_mode
        )

        # convert lh5_files into a nested list
        if isinstance(lh5_files, str):
            self.lh5_files = [expand_path(lh5_files, list=True, base_path=base_path)]
        elif not isinstance(lh5_files, Collection):
            msg = "lh5_files must be a string or list of strings"
            raise ValueError(msg)
        else:
            self.lh5_files = []
            for f in lh5_files:
                if isinstance(f, str):
                    self.lh5_files.append(
                        expand_path(f, list=True, base_path=base_path)
                    )
                elif isinstance(f, Collection) and all(
                    isinstance(name, str) for name in f
                ):
                    self.lh5_files.append(
                        [
                            path
                            for name in f
                            for path in expand_path(
                                name, list=True, base_path=base_path
                            )
                        ]
                    )
                else:
                    msg = "lh5_files must be a collection of strings with up to two levels of nesting"

        # convert groups into a nested list
        if isinstance(groups, str):
            self.groups = [[groups]] * len(self.lh5_files)
        elif not isinstance(groups, Collection):
            msg = "group must be a string or collection of strings"
            raise ValueError(msg)
        elif all(isinstance(g, str) for g in groups):
            self.groups = [groups] * len(self.lh5_files)
        else:
            self.groups = []
            for g in groups:
                if isinstance(g, str):
                    g = [g]  # noqa: PLW2901
                elif not isinstance(g, Collection) and all(
                    isinstance(name, str) for name in g
                ):
                    msg = "groups must be a collection of strings with up to two levels of nesting"
                    raise ValueError(msg)
                self.groups.append(g)

        # convert group_data into array of records
        if isinstance(group_data, pd.DataFrame):
            self.group_data = ak.Array(dict(group_data))
        elif group_data is not None:
            self.group_data = ak.Array(group_data)
        else:
            self.group_data = None

        if isinstance(self.group_data, ak.Array):
            if not self.group_data.fields:
                msg = "group_data must have named fields"
                raise ValueError(msg)
            self.group_data = ak.zip(
                {f: self.group_data[f] for f in self.group_data.fields}
            )
            if self.group_data.ndim == 1:
                self.group_data = ak.Array([self.group_data] * len(self.lh5_files))
            if not all(
                len(gd) == len(gps) for gd, gps in zip(self.group_data, self.groups)
            ):
                msg = "group_data must have same array structure as groups"
                raise ValueError(msg)

        # check that groups and lh5_files are compatible
        if len(self.groups) != len(self.lh5_files):
            msg = "lh5_files and groups must have same length at first level of nesting"
            raise ValueError(msg)

        # outer-product and flatten nested lists to get all group/file combinations for datasets
        file_list = []
        group_list = []
        group_data_list = [] if self.group_data is not None else None
        for i in range(len(self.lh5_files)):
            f_list = self.lh5_files[i]
            g_list = self.groups[i]
            gd_list = self.group_data[i] if self.group_data is not None else None
            for i_f, i_g in product(range(len(f_list)), range(len(g_list))):
                file_list.append(f_list[i_f])
                group_list.append(g_list[i_g])
                if gd_list is not None:
                    group_data_list.append(gd_list[i_g])
        self.lh5_files = file_list
        self.groups = group_list
        self.group_data = group_data_list
        self.n_datasets = len(self.lh5_files)

        if entry_list is not None and entry_mask is not None:
            msg = "entry_list and entry_mask arguments are mutually exclusive"
            raise ValueError(msg)

        # Map to last row in each file
        if ds_map is None:
            self.ds_map = np.full(self.n_datasets, np.iinfo("q").max, "q")
        else:
            self.ds_map = np.array(ds_map)

        # Map to last iterator entry for each file
        self.entry_map = np.full(self.n_datasets, np.iinfo("q").max, "q")

        self.friend = []
        self.friend_prefix = []
        self.friend_suffix = []

        if self.n_datasets == 0:
            msg = f"can't open any files from {lh5_files}"
            raise RuntimeError(msg)

        # lh5 buffer will contain all fields and be used for I/O (with a field mask)
        self.lh5_buffer = self.lh5_st.get_buffer(
            self.groups[0],
            self.lh5_files[0],
            size=0,
        )

        def get_available_fields(tab):
            ret = set()
            if not isinstance(tab, Table):
                return ret

            for k, v in tab.items():
                ret |= {k}
                if isinstance(v, Table):
                    ret |= {f"{k}/{field}" for field in get_available_fields(v)}
            return ret

        self.available_fields = get_available_fields(self.lh5_buffer)

        # set field mask and buffer length
        # Note: group_data is added to table here!
        self.reset_field_mask(field_mask)
        self.buffer_len = buffer_len

        # Attach the friend(s)
        if friend is None:
            friend = []
        elif isinstance(friend, LH5Iterator):
            friend = [friend]

        if len(friend) > 0:
            fr_buf_len = min(fr.buffer_len for fr in friend)
            self.buffer_len = min(self.buffer_len, fr_buf_len)

        if isinstance(friend_prefix, str):
            friend_prefix = [friend_prefix] * len(friend)
        if isinstance(friend_suffix, str):
            friend_suffix = [friend_suffix] * len(friend)
        for fr, prefix, suffix in zip(friend, friend_prefix, friend_suffix):
            self.add_friend(fr, prefix, suffix)

        self.i_start = i_start
        self.n_entries = n_entries
        self.current_i_entry = 0
        self.next_i_entry = 0

        # List of entry indices from each file
        self.local_entry_list = None
        self.global_entry_list = None
        if entry_list is not None:
            entry_list = list(entry_list)
            if isinstance(entry_list[0], int):
                self.local_entry_list = [None] * self.n_datasets
                self.global_entry_list = np.array(entry_list, "q")
                self.global_entry_list.sort()

            else:
                self.local_entry_list = [[]] * self.n_datasets
                for i_ds, local_list in enumerate(entry_list):
                    self.local_entry_list[i_ds] = np.array(local_list, "q")
                    self.local_entry_list[i_ds].sort()

        elif entry_mask is not None:
            # Convert entry mask into an entry list
            if isinstance(entry_mask, pd.Series):
                entry_mask = entry_mask.to_numpy()
            if isinstance(entry_mask, np.ndarray):
                self.local_entry_list = [None] * self.n_datasets
                self.global_entry_list = np.nonzero(entry_mask)[0]
            else:
                self.local_entry_list = [[]] * self.n_datasets
                for i_ds, local_mask in enumerate(entry_mask):
                    self.local_entry_list[i_ds] = np.nonzero(local_mask)[0]

    def _get_ds_cumlen(self, i_ds: int) -> int:
        """Helper to get cumulative dataset length of file/groups"""
        if i_ds < 0:
            return 0
        fcl = self.ds_map[i_ds]

        # if we haven't already calculated, calculate for all files up to i_ds
        if fcl == np.iinfo("q").max:
            i_start = np.searchsorted(self.ds_map, np.iinfo("q").max)
            fcl = self.ds_map[i_start - 1] if i_start > 0 else 0

            for i in range(i_start, i_ds + 1):
                fcl += self.lh5_st.read_n_rows(self.groups[i], self.lh5_files[i])
                self.ds_map[i] = fcl
        return fcl

    def _get_ds_cumentries(self, i_ds: int) -> int:
        """Helper to get cumulative iterator entries in file/groups"""
        if i_ds < 0:
            return 0
        n = self.entry_map[i_ds]

        # if we haven't already calculated, calculate for all files up to i_ds
        if n == np.iinfo("q").max:
            i_start = np.searchsorted(self.entry_map, np.iinfo("q").max)
            n = self.entry_map[i_start - 1] if i_start > 0 else 0

            for i in range(i_start, i_ds + 1):
                elist = self.get_ds_entrylist(i)
                fcl = self._get_ds_cumlen(i)
                if elist is None:
                    # no entry list provided
                    n = fcl
                else:
                    n += len(elist)
                    # check that file entries fall inside of file
                    if len(elist) > 0 and elist[-1] >= fcl:
                        logging.warning(f"Found entries out of range for file {i}")
                        n += np.searchsorted(elist, fcl, "right") - len(elist)
                self.entry_map[i] = n
        return n

    def get_ds_entrylist(self, i_ds: int) -> np.ndarray:
        """Helper to get entry list for dataset"""
        # If no entry list is provided
        if self.local_entry_list is None:
            return None

        elist = self.local_entry_list[i_ds]
        if elist is None:
            # Get local entrylist for this dataset from global entry list
            f_start = self._get_ds_cumlen(i_ds - 1)
            f_end = self._get_ds_cumlen(i_ds)
            i_start = self._get_ds_cumentries(i_ds - 1)
            i_stop = np.searchsorted(self.global_entry_list, f_end, "right")
            elist = np.array(self.global_entry_list[i_start:i_stop], "q") - f_start
            self.local_entry_list[i_ds] = elist
        return elist

    def get_global_entrylist(self) -> np.ndarray:
        """Get global entry list, constructing it if needed"""
        if self.global_entry_list is None and self.local_entry_list is not None:
            self.global_entry_list = np.zeros(len(self), "q")
            for i_ds in range(self.n_datasets):
                i_start = self._get_ds_cumentries(i_ds - 1)
                i_stop = self._get_ds_cumentries(i_ds)
                f_start = self._get_ds_cumlen(i_ds - 1)
                self.global_entry_list[i_start:i_stop] = (
                    self.get_ds_entrylist(i_ds) + f_start
                )
        return self.global_entry_list

    def read(self, i_entry: int, n_entries: int | None = None) -> Table:
        "Read the nextlocal chunk of events, starting at entry."
        self.lh5_buffer.resize(0)

        if n_entries is None:
            n_entries = self.buffer_len
        elif n_entries == 0:
            return self.lh5_buffer
        elif n_entries > self.buffer_len:
            msg = "n_entries cannot be larger than buffer_len"
            raise ValueError(msg)

        # if dataset hasn't been opened yet, search through datasets
        # sequentially until we find the right one
        i_ds = np.searchsorted(self.entry_map, i_entry, "right")
        if i_ds < self.n_datasets and self.entry_map[i_ds] == np.iinfo("q").max:
            while i_ds < self.n_datasets and i_entry >= self._get_ds_cumentries(i_ds):
                i_ds += 1

        if i_ds == self.n_datasets:
            return self.lh5_buffer
        local_i_entry = i_entry - self._get_ds_cumentries(i_ds - 1)

        while len(self.lh5_buffer) < n_entries and i_ds < self.n_datasets:
            # Loop through datasets
            local_idx = self.get_ds_entrylist(i_ds)
            if local_idx is not None and len(local_idx) == 0:
                i_ds += 1
                local_i_entry = 0
                continue

            i_local = local_i_entry if local_idx is None else local_idx[local_i_entry]
            self.lh5_buffer = self.lh5_st.read(
                self.groups[i_ds],
                self.lh5_files[i_ds],
                start_row=i_local,
                n_rows=n_entries - len(self.lh5_buffer),
                idx=local_idx,
                field_mask=self.field_mask,
                obj_buf=self.lh5_buffer,
                obj_buf_start=len(self.lh5_buffer),
            )

            if self.group_data is not None:
                data = self.group_data[i_ds]
                for f in data.fields:
                    self.lh5_buffer[f][local_idx:] = data[f]

            i_ds += 1
            local_i_entry = 0

        self.current_i_entry = i_entry

        for friend in self.friend:
            friend.read(i_entry)

        return self.lh5_buffer

    @property
    def buffer_len(self):
        return self._buffer_len

    @buffer_len.setter
    def buffer_len(self, buffer_len: str | ureg.Quantity | int):
        if isinstance(buffer_len, str):
            buffer_len = ureg.Quantity(buffer_len)
        if isinstance(buffer_len, ureg.Quantity):
            f = self.lh5_files[0]
            g = self.groups[0]
            buffer_len = int(
                buffer_len
                / (self.lh5_st.read_size_in_bytes(g, f) * ureg.B)
                * self.lh5_st.read_n_rows(g, f)
            )

        self._buffer_len = buffer_len
        for fr in self.friend:
            fr.buffer_len = buffer_len

    def add_friend(self, friend: LH5Iterator, prefix: str = "", suffix: str = ""):
        """Add a friend which will be iterated alongside this, returning a Table
        joining the contents of each.

        Parameters
        ----------
        friend
            LH5Iterator to be friended to this one
        prefix
            string prepended to field names; useful for disambiguating conflicts
        suffix
            string appended to field names; useful for disambiguating conflicts
        """
        if not isinstance(friend, LH5Iterator):
            msg = "Friend must be an LH5Iterator"
            raise ValueError(msg)

        # set buffer_lens to be equal
        if friend.buffer_len > self.buffer_len:
            friend.buffer_len = self.buffer_len
        elif friend.buffer_len < self.buffer_len:
            self.buffer_len = friend.buffer_len
        friend.lh5_buffer.resize(len(self.lh5_buffer))

        self.friend += [friend]
        self.friend_prefix += [prefix]
        self.friend_suffix += [suffix]
        self.lh5_buffer.join(
            friend.lh5_buffer,
            keep_mine=True,
            prefix=prefix,
            suffix=suffix,
        )

    def reset_field_mask(
        self,
        mask: Collection[str]
        | Mapping[str, bool]
        | Collection[Collection[str]]
        | Collection[Mapping[str, bool]]
        | None,
        warn_missing=True,
    ):
        """Replaces the field mask of this iterator and any friends with mask.

        - If ``None``, set this and all friends to have no mask.
        - If a collection of strings or mapping from strings to bools, set the mask
          for this and all friends; in the case of a conflict, use first column found. If a
          prefix or suffix is included for the friend, it must be included in this mask
        - If a collection of collections, use the first item to set this mask, and subsequent
          items to set friend masks. In this case, do not include prefixes or suffixes in names
        """
        if mask is None:
            self.field_mask = self.available_fields

            for fr in self.friend:
                fr.reset_field_mask(None)

            remaining_fields = []

        elif isinstance(mask, Mapping):
            self.field_mask = {
                field: mask[field] for field in self.available_fields if field in mask
            }
            mask = {
                field: mask[field] for field in mask if field not in self.field_mask
            }

            for fr, pre, suf in zip(
                self.friend, self.friend_prefix, self.friend_suffix
            ):
                mask_lookup = {
                    f"{pre}{field}{suf}": field for field in fr.available_fields
                }
                fr_mask = {
                    mask_lookup[field]: mask[field]
                    for field in mask_lookup
                    if field in mask
                }
                fr.reset_field_mask(fr_mask)
                mask = {
                    field: mask[field] for field in mask if field not in mask_lookup
                }

            remaining_fields = mask

        elif isinstance(mask, Collection) and all(isinstance(m, str) for m in mask):
            mask = set(mask)
            self.field_mask = mask & set(self.available_fields)
            mask -= self.field_mask

            for fr, pre, suf in zip(
                self.friend, self.friend_prefix, self.friend_suffix
            ):
                mask_lookup = {
                    f"{pre}{field}{suf}": field for field in fr.available_fields
                }
                fr_mask = {mask_lookup[field] for field in mask if field in mask_lookup}
                fr.reset_field_mask(fr_mask)
                mask -= set(mask_lookup)

            remaining_fields = mask

        # Create a new buffer and move any elements from the old into the new
        def copy_data(old_buffer, new_buffer):
            if isinstance(new_buffer, Table):
                for k, v in new_buffer.items():
                    if k in old_buffer:
                        new_buffer[k] = copy_data(v, old_buffer[k])
                return new_buffer
            return old_buffer

        self.lh5_buffer = copy_data(
            self.lh5_buffer,
            self.lh5_st.get_buffer(
                self.groups[0],
                self.lh5_files[0],
                size=len(self.lh5_buffer),
                field_mask=self.field_mask,
            ),
        )

        for fr, pre, suf in zip(self.friend, self.friend_prefix, self.friend_suffix):
            self.lh5_buffer.join(
                fr.lh5_buffer,
                keep_mine=True,
                prefix=pre,
                suffix=suf,
            )

        # join Table with group metadata; repeat first record to initialize
        if self.group_data:
            tb_gd = deepcopy(Table(ak.Array([self.group_data[0]])))
            tb_gd.resize(len(self.lh5_buffer))
            self.lh5_buffer.join(tb_gd)

        if warn_missing and len(remaining_fields) > 0:
            logging.warning(f"Fields {remaining_fields} in field mask were not found")

    @property
    def current_local_entries(self) -> NDArray[int]:
        """Return list of local dataset entries in buffer"""
        cur_entries = np.zeros(len(self.lh5_buffer), dtype="int32")
        i_ds = np.searchsorted(self.entry_map, self.current_i_entry, "right")
        ds_start = self._get_ds_cumentries(i_ds - 1)
        i_local = self.current_i_entry - ds_start
        i = 0

        while i < len(cur_entries):
            # number of entries to read from this file
            ds_end = self._get_ds_cumentries(i_ds)
            n = min(ds_end - ds_start - i_local, len(cur_entries) - i)
            entries = self.get_ds_entrylist(i_ds)

            if entries is None:
                cur_entries[i : i + n] = np.arange(i_local, i_local + n)
            else:
                cur_entries[i : i + n] = entries[i_local : i_local + n]

            i_ds += 1
            ds_start = ds_end
            i_local = 0
            i += n

        return cur_entries

    @property
    def current_global_entries(self) -> NDArray[int]:
        """Return list of local file entries in buffer"""
        cur_entries = np.zeros(len(self.lh5_buffer), dtype="int32")
        i_ds = np.searchsorted(self.entry_map, self.current_i_entry, "right")
        ds_start = self._get_ds_cumentries(i_ds - 1)
        i_local = self.current_i_entry - ds_start
        i = 0

        while i < len(cur_entries):
            # number of entries to read from this file
            ds_end = self._get_ds_cumentries(i_ds)
            n = min(ds_end - ds_start - i_local, len(cur_entries) - i)
            entries = self.get_ds_entrylist(i_ds)

            if entries is None:
                cur_entries[i : i + n] = self._get_ds_cumlen(i_ds - 1) + np.arange(
                    i_local, i_local + n
                )
            else:
                cur_entries[i : i + n] = (
                    self._get_ds_cumlen(i_ds - 1) + entries[i_local : i_local + n]
                )

            i_ds += 1
            ds_start = ds_end
            i_local = 0
            i += n

        return cur_entries

    @property
    def current_files(self) -> NDArray[str]:
        """Return list of file names for entries in buffer"""
        cur_files = np.zeros(len(self.lh5_buffer), dtype=np.dtypes.StringDType)
        i_ds = np.searchsorted(self.entry_map, self.current_i_entry, "right")
        ds_start = self._get_ds_cumentries(i_ds - 1)
        i_local = self.current_i_entry - ds_start
        i = 0

        while i < len(cur_files):
            # number of entries to read from this file
            ds_end = self._get_ds_cumentries(i_ds)
            n = min(ds_end - ds_start - i_local, len(cur_files) - i)
            cur_files[i : i + n] = self.lh5_files[i_ds]

            i_ds += 1
            ds_start = ds_end
            i_local = 0
            i += n

        return cur_files

    @property
    def current_groups(self) -> NDArray[str]:
        """Return list of group names for entries in buffer"""
        cur_groups = np.zeros(len(self.lh5_buffer), dtype=np.dtypes.StringDType)
        i_ds = np.searchsorted(self.entry_map, self.current_i_entry, "right")
        ds_start = self._get_ds_cumentries(i_ds - 1)
        i_local = self.current_i_entry - ds_start
        i = 0

        while i < len(cur_groups):
            # number of entries to read from this file
            ds_end = self._get_ds_cumentries(i_ds)
            n = min(ds_end - ds_start - i_local, len(cur_groups) - i)
            cur_groups[i : i + n] = self.groups[i_ds]

            i_ds += 1
            ds_start = ds_end
            i_local = 0
            i += n

        return cur_groups

    def __len__(self) -> int:
        """Return the total number of entries to be read."""
        if len(self.entry_map) == 0:
            return 0
        if self.n_entries is None:
            return self._get_ds_cumentries(self.n_datasets - 1)
        # only check as many files as we strictly need to
        for i in range(self.n_datasets):
            if self.n_entries < self._get_ds_cumentries(i):
                return self.n_entries
        return self._get_ds_cumentries(self.n_datasets - 1)

    def __iter__(self) -> LH5Iterator:
        """Loop through entries in blocks of size buffer_len."""
        self.current_i_entry = 0
        self.next_i_entry = self.i_start
        return self

    def __next__(self) -> Table:
        """Read next buffer_len entries and return lh5_table and iterator entry."""
        n_entries = self.n_entries
        if n_entries is not None:
            n_entries = min(
                self.buffer_len, n_entries + self.i_start - self.next_i_entry
            )

        buf = self.read(self.next_i_entry, n_entries)
        if len(buf) == 0:
            raise StopIteration
        self.next_i_entry = self.current_i_entry + len(buf)
        return buf

    def __deepcopy__(self, memo):
        """Deep copy everything except lh5_st"""
        result = LH5Iterator.__new__(LH5Iterator)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "lh5_st":
                result.lh5_st = LH5Store(
                    base_path=self.lh5_st.base_path, keep_open=self.lh5_st.keep_open
                )
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    def __getstate__(self):
        """Do not try to pickle lh5_st or lh5_buffer"""
        return dict(
            self.__dict__,
            lh5_st={
                "base_path": self.lh5_st.base_path,
                "keep_open": self.lh5_st.keep_open,
            },
            lh5_buffer=None,
        )

    def __setstate__(self, d):
        """Reinitialize lh5_st and lh5_buffer to avoid potential issues"""
        self.__dict__ = d
        self.lh5_st = LH5Store(**(d["lh5_st"]))
        self.lh5_st.gimme_file(self.lh5_files[0])
        self.lh5_buffer = self.lh5_st.get_buffer(
            self.groups[0],
            self.lh5_files[0],
            size=self.buffer_len,
            field_mask=self.field_mask,
        )
        for fr, pre, suf in zip(self.friend, self.friend_prefix, self.friend_suffix):
            self.lh5_buffer.join(
                fr.lh5_buffer,
                keep_mine=True,
                prefix=pre,
                suffix=suf,
            )

    def _select_groups(self, i_beg, i_end):
        """Reduce list of files and groups; used by _generate_workers"""
        s = slice(i_beg, i_end)
        self.lh5_files = self.lh5_files[s]
        self.groups = self.groups[s]
        self.n_datasets = i_end - i_beg

        if i_beg > 0:
            np.subtract(
                self.ds_map,
                self.ds_map[i_beg - 1],
                out=self.ds_map,
                where=self.ds_map != np.iinfo("q").max,
            )
            np.subtract(
                self.entry_map,
                self.entry_map[i_beg - 1],
                out=self.entry_map,
                where=self.entry_map != np.iinfo("q").max,
            )
        self.ds_map = self.ds_map[s]
        self.entry_map = self.entry_map[s]

        if self.local_entry_list is not None:
            self.local_entry_list = self.local_entry_list[s]
        self.global_entry_list = None

        for fr in self.friend:
            fr._select_groups(i_beg, i_end)

    def _generate_workers(self, n_workers: int):
        """Create n_workers copy of this iterator, dividing the datasets (file/groups)
        groups between them. These are intended for parallel use"""
        i_datasets = np.linspace(0, len(self.lh5_files), n_workers + 1).astype("int")
        # if we have an entry list, get local entries for all files
        if self.local_entry_list is not None:
            for i in range(len(self.lh5_files)):
                self.get_ds_entrylist(i)

        worker_its = []
        for i_worker in range(n_workers):
            it = deepcopy(self)
            it._select_groups(i_datasets[i_worker], i_datasets[i_worker + 1])
            worker_its += [it]

        return worker_its

    def map(
        self,
        fun: Callable[Table, LH5Iterator, Any],
        aggregate: Callable = None,
        init: Any = None,
        begin: Callable[LH5Iterator] = None,
        terminate: Callable[LH5Iterator] = None,
        processes: int = None,
        executor: Executor = None,
    ) -> Iterator[Any]:
        """Map function over iterator blocks.

        Returns order-preserving list of outputs. Can be multi-threaded
        provided there are no attempts to modify existing objects.
        Multi-threading splits the iterator into multiple independent
        streams with an approximately equal number of files/groups,
        concurrently processed under a single program multiple data
        model. Results will be returned asynchronously for each process.

        Note: see :meth:`query` and :meth:`hist`

        Example
        -------
        Process a table and sum the products at the end::

            def process(lh5_tab, lh5_it):
                ...process the table
                return result_of_processing

            results = lh5_it.map(process, processes=4)
            # results are an iterator over lists
            result = sum(val for result in results for val in result)

        Process a table as above, using aggregate to sum the results::

            def process(lh5_tab, lh5_it):
                ...process the table
                return result_of_processing

            result = lh5_it.map(process, aggregate=np.add, init=0, processes=4)

        Process a table using a more arbitrary output::

            class Result:
                def __init__(self):
                    ...initialize

                @classmethod
                def process_table(tab):
                    ...process the table

                def aggregate(self, result):
                    ...add data from processing into object

            result = lh5_it.map(
                Result.process_table,
                aggregate=Result.aggregate,
                init=Result(),
                processes=4
            )

        Parameters
        ----------
        fun:
            function with signature ``fun(lh5_obj: Table, it: LH5Iterator) -> Any``
            Outputs of function will be collected in list and returned
        aggregate:
            function used to iterably combine outputs of ``fun`` for each block
            of data. Should have two inputs; first input should be the type of
            the aggregate, and second of the type returned by ``fun``. This function
            can either return the result, or perform the aggregation in-place on
            the first element and return ``None``. If using multi-processing, ``map``
            will return an async-iterator over the aggregated results from each process.
            If ``None``, do not aggregate and instead return will iterate over
            result for each block.
        init:
            initial value used for aggregation. If using an aggregating function
            and ``init`` is ``None``, perform a deep copy of the first element
        begin:
            function with signature ``fun(it: LH5Iterator)`` that is run before we
            loop through a chunk of the iterator
        terminate:
            function with the signature ``fun(it: LH5Iterator)`` that is run after
            we finish looping through a chunk of the iterator
        processes:
            number of processes. If ``None``, use number equal to threads available
            to ``executor`` (if provided), or else do not parallelize
        executor:
            :class:`concurrent.futures.Executor` object for managing parallelism.
            If ``None``, create a :class:`concurrent.futures.`ProcessPoolExecutor`
            with number of processes equal to ``processes``.
        """

        # if no aggregate is provided, append results to a list
        if aggregate is None:
            init = []
            aggregate = _append_copy

        if processes is None:
            if isinstance(executor, Executor):
                processes = executor._max_workers
            else:
                return _map_helper(fun, aggregate, init, begin, terminate, self)

        if executor is None:
            executor = ProcessPoolExecutor(processes)

        it_pool = self._generate_workers(processes)

        result = executor.map(
            partial(_map_helper, fun, aggregate, init, begin, terminate), it_pool
        )

        # If no aggregator was given, chain iterators
        if aggregate is _append_copy:
            return chain.from_iterable(result)
        return result

    def query(
        self,
        where: Callable | str,
        processes: Executor | int = None,
        executor: Executor = None,
        library: str = None,
    ):
        """
        Query the data files in the iterator

        Returns the selected data as a single table in one of several formats.

        Examples
        --------
        Query data using a string selection::

            tab = lh5_it.query("(col1 == 0) & (col2 > 100)")

        Query data using a function::

            def select(lh5_tab, lh5_it):
                ...process data and produce a new table
                return result

            tab = lh5_it.query(select)

        Parameters
        ----------
        where:
            A filter function for selecting data entries. Can be:

            - A function that returns reduced data, with signature
              ``fun(lh5_obj: Table, it: LH5Iterator)``. Can return:

              - :class:`numpy.ndarray`: if 1D list of values; if 2D list of lists of
                values in same order as axes
              - ``Collection[ArrayLike]``: return list of values in same order as axes
              - ``Mapping[str, ArrayLike]``: mapping from axis name to values
              - ``pandas.DataFrame``: pandas dataframe. Treat as mapping from column
                name to values

            - A string expression. This will call `Table.eval` to select events and
                return a table with all fields in the `field_mask`, in a data format
                provided with `library`.

        processes:
            number of processes. If ``None``, use number equal to threads available
            to ``executor`` (if provided), or else do not parallelize
        executor:
            :class:`concurrent.futures.Executor` object for managing parallelism.
            If ``None``, create a :class:`concurrent.futures.`ProcessPoolExecutor`
            with number of processes equal to ``processes``.
        library:
            library to convert the columns to when using a string expression for ``where``.
            See :meth:`Table.eval`.
        """
        if where is None:
            where = _identity

        if isinstance(where, str):
            where = _table_query(where, library)

        test = where(self.lh5_buffer, self)
        if isinstance(test, LGDOCollection):
            it = self.map(
                where, processes=processes, executor=executor, aggregate=Table.append
            )
            if isinstance(it, LGDOCollection):
                return it
            ret = next(it)
            for res in it:
                ret.append(res)
            return ret
        if isinstance(test, pd.DataFrame):
            return pd.concat(
                list(self.map(where, processes=processes, executor=executor)),
                ignore_index=True,
            )
        if isinstance(test, np.ndarray):
            return np.concatenate(
                list(self.map(where, processes=processes, executor=executor))
            )
        if isinstance(test, ak.Array):
            return ak.concatenate(
                list(self.map(where, processes=processes, executor=executor))
            )

        msg = f"Cannot call query with return type {test.__class__}. "
        "Allowed return types: LGDOCollection, np.array, pd.DataFrame, ak.Array"
        raise ValueError(msg)

    def hist(
        self,
        ax: Hist | axis | Collection[axis],
        where: Callable | str = None,
        keys: Collection[str] | str = None,
        processes: Executor | int = None,
        executor: Executor = None,
        **hist_kwargs,
    ) -> Hist:
        """
        Fill a histogram from data produced by a query selecting on ``where``. If
        ``where`` is ``None``, fill with all data fetched by iterator.

        Examples
        --------
        Build a 1D histogram of values in ``col3`` with a string selection::

            h = lh5_it.query(
                hist.axis.RegularAxis(100, 0, 500, label="col3")
                where = "(col1 == 0) & (col2 > 100)",
                keys = "col3"
            )

        Build a 2D histogram with a value axis and string-category axis after
        applying some processing::

            def get_val(lh5_tab, lh5_it):
                ...process data
                return value, category

            h = lh5_it
                [hist.axis.RegularAxis(100, 0, 500, label="Value"),
                 hist.axis.StrCategory([], growth=True, label="Category")],
                 where = get_val,
            )

        Parameters
        ----------
        ax:
            :class:`hist.axis` object(s) used to construct the histogram. Can provide a
            :class:`hist.Hist` which will be filled as well.
        where:
            A filter function for selecting data entries to put into the histogram. Can be:

            - A function that returns reduced data, with signature
              ``fun(lh5_obj: Table, it: LH5Iterator)``. Can return:

              - :class:`numpy.ndarray`: if 1D list of values; if 2D list of lists of
                values in same order as axes
              - ``Collection[ArrayLike]``: return list of values in same order as axes
              - ``Mapping[str, ArrayLike]``: mapping from axis name to values
              - :class:`pandas.DataFrame`: treat as mapping from column name to values
            - A string expression. This will call :meth:`lgdo.Table.eval` and return
              a pandas DataFrame containing all columns in the fields mask.

        keys:
            list of keys fields corresponding to axes. Use if where
            returns a mapping with names different from axis names.
        processes:
            number of processes. If ``None``, use number equal to threads available
            to ``executor`` (if provided), or else do not parallelize
        executor:
            :class:`concurrent.futures.Executor` object for managing parallelism.
            If ``None``, create a :class:`concurrent.futures.`ProcessPoolExecutor`
            with number of processes equal to ``processes``.
        hist_kwargs:
            additional keyword arguments for constructing Hist. See :class:`hist.Hist`.
        """

        # get initial hist for each thread
        if isinstance(ax, axis.AxesMixin):
            h = Hist(ax, **hist_kwargs)
        elif isinstance(ax, Collection):
            h = Hist(*ax, **hist_kwargs)
        elif isinstance(ax, Hist):
            h = ax.copy()
            h[...] = 0

        if where is None:
            where = _identity
        elif isinstance(where, str):
            where = _table_query(where, "ak")

        h = self.map(
            where,
            processes=processes,
            executor=executor,
            aggregate=_hist_filler(keys),
            init=h,
        )
        if isinstance(h, Iterator):
            h = sum(h)

        if isinstance(ax, Hist):
            ax += h
            return ax
        return h


# Would that python multiprocessing allowed lambdas...
def _identity(val, _):
    return val


def _append_copy(list, val):
    "Helper for aggregating tables in query"
    list.append(deepcopy(val))


def _map_helper(fun, aggregator, init, begin, terminate, it):
    "Helper for executing int, begin and terminate functions when calling map"
    if begin:
        begin(it)

    aggregate = init
    for tab in it:
        result = fun(tab, it)

        if aggregate is None:
            # if no init, initialize on first entry
            aggregate = deepcopy(result)
        else:
            res = aggregator(aggregate, result)
            if res is not None:
                aggregate = res

    if terminate:
        terminate(it)

    return aggregate


@dataclass
class _table_query:
    "Helper for when query is called on a string"

    expr: str
    library: str

    def __call__(self, tab, _):
        "Evaluate selection and return selected elements"
        # Note this could probably be done more simply if we
        # implemented __getitem__ to work with lists/slices
        mask = tab.eval(self.expr)
        ret = tab.view_as("ak")[mask]
        if self.library == "ak":
            return ret
        if self.library is None:
            return Table(ret)
        return Table(ret).view_as(self.library)


class _hist_filler:
    "Helper for filling histogram"

    def __init__(self, keys):
        if keys is not None:
            if isinstance(keys, str):
                keys = [keys]
            elif not isinstance(keys, list):
                keys = list(keys)
        self.keys = keys

    def __call__(self, hist, data):
        if isinstance(data, np.ndarray) and len(data.shape) == 1:
            hist.fill(data)
        elif isinstance(data, np.ndarray) and len(data.shape) == 2:
            hist.fill(*data)
        elif isinstance(data, (pd.DataFrame, Mapping)):
            if self.keys is not None:
                hist.fill(*[data[k] for k in self.keys])
            else:
                hist.fill(**data)
        elif isinstance(data, ak.Array):
            # TODO: handle nested ak arrays
            if self.keys is not None:
                hist.fill(*[data[k] for k in self.keys])
            else:
                hist.fill(*[data[f] for f in data.fields])
        elif isinstance(data, Collection):
            hist.fill(*data)
        else:
            msg = "data returned by where is not compatible with hist. Must be a 1d or 2d numpy array, a list of arrays, or a mapping from str to array"
            raise ValueError(msg)
