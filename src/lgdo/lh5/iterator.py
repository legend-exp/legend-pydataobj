from __future__ import annotations

import logging
import typing

import numpy as np
import pandas as pd

from ..types import Array, Scalar, Struct, VectorOfVectors
from .store import LH5Store
from .utils import expand_path

LGDO = typing.Union[Array, Scalar, Struct, VectorOfVectors]


class LH5Iterator(typing.Iterator):
    """
    A class for iterating through one or more LH5 files, one block of entries
    at a time. This also accepts an entry list/mask to enable event selection,
    and a field mask.

    This class can be used either for random access:

    >>> lh5_obj, n_rows = lh5_it.read(entry)

    to read the block of entries starting at entry. In case of multiple files
    or the use of an event selection, entry refers to a global event index
    across files and does not count events that are excluded by the selection.

    This can also be used as an iterator:

    >>> for lh5_obj, entry, n_rows in LH5Iterator(...):
    >>>    # do the thing!

    This is intended for if you are reading a large quantity of data but
    want to limit your memory usage (particularly when reading in waveforms!).
    The ``lh5_obj`` that is read by this class is reused in order to avoid
    reallocation of memory; this means that if you want to hold on to data
    between reads, you will have to copy it somewhere!
    """

    def __init__(
        self,
        lh5_files: str | list[str],
        groups: str | list[str],
        base_path: str = "",
        entry_list: list[int] | list[list[int]] | None = None,
        entry_mask: list[bool] | list[list[bool]] | None = None,
        field_mask: dict[str, bool] | list[str] | tuple[str] | None = None,
        buffer_len: int = 3200,
        friend: typing.Iterator | None = None,
    ) -> None:
        """
        Parameters
        ----------
        lh5_files
            file or files to read from. May include wildcards and environment
            variables.
        groups
            HDF5 group(s) to read. If a list is provided for both lh5_files
            and group, they must be the same size. If a file is wild-carded,
            the same group will be assigned to each file found
        entry_list
            list of entry numbers to read. If a nested list is provided,
            expect one top-level list for each file, containing a list of
            local entries. If a list of ints is provided, use global entries.
        entry_mask
            mask of entries to read. If a list of arrays is provided, expect
            one for each file. Ignore if a selection list is provided.
        field_mask
            mask of which fields to read. See :meth:`LH5Store.read` for
            more details.
        buffer_len
            number of entries to read at a time while iterating through files.
        friend
            a \"friend\" LH5Iterator that will be read in parallel with this.
            The friend should have the same length and entry list. A single
            LH5 table containing columns from both iterators will be returned.
        """
        self.lh5_st = LH5Store(base_path=base_path, keep_open=True)

        # List of files, with wildcards and env vars expanded
        if isinstance(lh5_files, str):
            lh5_files = [lh5_files]
            if isinstance(groups, list):
                lh5_files *= len(groups)
        elif not isinstance(lh5_files, list):
            msg = "lh5_files must be a string or list of strings"
            raise ValueError(msg)

        if isinstance(groups, str):
            groups = [groups] * len(lh5_files)
        elif not isinstance(groups, list):
            msg = "group must be a string or list of strings"
            raise ValueError(msg)

        if len(groups) != len(lh5_files):
            msg = "lh5_files and groups must have same length"
            raise ValueError(msg)

        self.lh5_files = []
        self.groups = []
        for f, g in zip(lh5_files, groups):
            f_exp = expand_path(f, list=True, base_path=base_path)
            self.lh5_files += f_exp
            self.groups += [g] * len(f_exp)

        if entry_list is not None and entry_mask is not None:
            msg = "entry_list and entry_mask arguments are mutually exclusive"
            raise ValueError(msg)

        # Map to last row in each file
        self.file_map = np.full(len(self.lh5_files), np.iinfo("i").max, "i")
        # Map to last iterator entry for each file
        self.entry_map = np.full(len(self.lh5_files), np.iinfo("i").max, "i")
        self.buffer_len = buffer_len

        if len(self.lh5_files) > 0:
            f = self.lh5_files[0]
            g = self.groups[0]
            self.lh5_buffer = self.lh5_st.get_buffer(
                g,
                f,
                size=self.buffer_len,
                field_mask=field_mask,
            )
            self.file_map[0] = self.lh5_st.read_n_rows(g, f)
        else:
            msg = f"can't open any files from {lh5_files}"
            raise RuntimeError(msg)

        self.n_rows = 0
        self.current_entry = 0
        self.next_entry = 0

        self.field_mask = field_mask

        # List of entry indices from each file
        self.local_entry_list = None
        self.global_entry_list = None
        if entry_list is not None:
            entry_list = list(entry_list)
            if isinstance(entry_list[0], int):
                self.local_entry_list = [None] * len(self.file_map)
                self.global_entry_list = np.array(entry_list, "i")
                self.global_entry_list.sort()

            else:
                self.local_entry_list = [[]] * len(self.file_map)
                for i_file, local_list in enumerate(entry_list):
                    self.local_entry_list[i_file] = np.array(local_list, "i")
                    self.local_entry_list[i_file].sort()

        elif entry_mask is not None:
            # Convert entry mask into an entry list
            if isinstance(entry_mask, pd.Series):
                entry_mask = entry_mask.to_numpy()
            if isinstance(entry_mask, np.ndarray):
                self.local_entry_list = [None] * len(self.file_map)
                self.global_entry_list = np.nonzero(entry_mask)[0]
            else:
                self.local_entry_list = [[]] * len(self.file_map)
                for i_file, local_mask in enumerate(entry_mask):
                    self.local_entry_list[i_file] = np.nonzero(local_mask)[0]

        # Attach the friend
        if friend is not None:
            if not isinstance(friend, typing.Iterator):
                msg = "Friend must be an Iterator"
                raise ValueError(msg)
            self.lh5_buffer.join(friend.lh5_buffer)
        self.friend = friend

    def _get_file_cumlen(self, i_file: int) -> int:
        """Helper to get cumulative file length of file"""
        if i_file < 0:
            return 0
        fcl = self.file_map[i_file]
        if fcl == np.iinfo("i").max:
            fcl = self._get_file_cumlen(i_file - 1) + self.lh5_st.read_n_rows(
                self.groups[i_file], self.lh5_files[i_file]
            )
            self.file_map[i_file] = fcl
        return fcl

    def _get_file_cumentries(self, i_file: int) -> int:
        """Helper to get cumulative iterator entries in file"""
        if i_file < 0:
            return 0
        n = self.entry_map[i_file]
        if n == np.iinfo("i").max:
            elist = self.get_file_entrylist(i_file)
            fcl = self._get_file_cumlen(i_file)
            if elist is None:
                # no entry list provided
                n = fcl
            else:
                file_entries = self.get_file_entrylist(i_file)
                n = len(file_entries)
                # check that file entries fall inside of file
                if n > 0 and file_entries[-1] >= fcl:
                    logging.warning(f"Found entries out of range for file {i_file}")
                    n = np.searchsorted(file_entries, fcl, "right")
                n += self._get_file_cumentries(i_file - 1)
            self.entry_map[i_file] = n
        return n

    def get_file_entrylist(self, i_file: int) -> np.ndarray:
        """Helper to get entry list for file"""
        # If no entry list is provided
        if self.local_entry_list is None:
            return None

        elist = self.local_entry_list[i_file]
        if elist is None:
            # Get local entrylist for this file from global entry list
            f_start = self._get_file_cumlen(i_file - 1)
            f_end = self._get_file_cumlen(i_file)
            i_start = self._get_file_cumentries(i_file - 1)
            i_stop = np.searchsorted(self.global_entry_list, f_end, "right")
            elist = np.array(self.global_entry_list[i_start:i_stop], "i") - f_start
            self.local_entry_list[i_file] = elist
        return elist

    def get_global_entrylist(self) -> np.ndarray:
        """Get global entry list, constructing it if needed"""
        if self.global_entry_list is None and self.local_entry_list is not None:
            self.global_entry_list = np.zeros(len(self), "i")
            for i_file in range(len(self.lh5_files)):
                i_start = self.get_file_cumentries(i_file - 1)
                i_stop = self.get_file_cumentries(i_file)
                f_start = self.get_file_cumlen(i_file - 1)
                self.global_entry_list[i_start:i_stop] = (
                    self.get_file_entrylist(i_file) + f_start
                )
        return self.global_entry_list

    def read(self, entry: int) -> tuple[LGDO, int]:
        """Read the nextlocal chunk of events, starting at entry. Return the
        LH5 buffer and number of rows read."""
        self.n_rows = 0
        i_file = np.searchsorted(self.entry_map, entry, "right")

        # if file hasn't been opened yet, search through files
        # sequentially until we find the right one
        if i_file < len(self.lh5_files) and self.entry_map[i_file] == np.iinfo("i").max:
            while i_file < len(self.lh5_files) and entry >= self._get_file_cumentries(
                i_file
            ):
                i_file += 1

        if i_file == len(self.lh5_files):
            return (self.lh5_buffer, self.n_rows)
        local_entry = entry - self._get_file_cumentries(i_file - 1)

        while self.n_rows < self.buffer_len and i_file < len(self.file_map):
            # Loop through files
            local_idx = self.get_file_entrylist(i_file)
            if local_idx is not None and len(local_idx) == 0:
                i_file += 1
                local_entry = 0
                continue

            i_local = local_idx[local_entry] if local_idx is not None else local_entry
            self.lh5_buffer, n_rows = self.lh5_st.read(
                self.groups[i_file],
                self.lh5_files[i_file],
                start_row=i_local,
                n_rows=self.buffer_len - self.n_rows,
                idx=local_idx,
                field_mask=self.field_mask,
                obj_buf=self.lh5_buffer,
                obj_buf_start=self.n_rows,
            )

            self.n_rows += n_rows
            i_file += 1
            local_entry = 0

        self.current_entry = entry

        if self.friend is not None:
            self.friend.read(entry)

        return (self.lh5_buffer, self.n_rows)

    def reset_field_mask(self, mask):
        """Replaces the field mask of this iterator and any friends with mask"""
        self.field_mask = mask
        if self.friend is not None:
            self.friend.reset_field_mask(mask)

    def __len__(self) -> int:
        """Return the total number of entries."""
        return (
            self._get_file_cumentries(len(self.lh5_files) - 1)
            if len(self.entry_map) > 0
            else 0
        )

    def __iter__(self) -> typing.Iterator:
        """Loop through entries in blocks of size buffer_len."""
        self.current_entry = 0
        self.next_entry = 0
        return self

    def __next__(self) -> tuple[LGDO, int, int]:
        """Read next buffer_len entries and return lh5_table, iterator entry
        and n_rows read."""
        buf, n_rows = self.read(self.next_entry)
        self.next_entry = self.current_entry + n_rows
        if n_rows == 0:
            raise StopIteration
        return (buf, self.current_entry, n_rows)
