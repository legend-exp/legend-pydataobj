"""
Implements a LEGEND Data Object representing an array of detector IDs.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np

from .array import Array


class ArrayOfDetectorIDs(Array):
    r"""
    Array of detector IDs, which are uint32 values encoding the name
    of a detector in the LEGEND experiment. See
    `Detector ID Encoding <https://legend-exp.github.io/legend-data-format-specs/dev/detector_ids/#Detector-ID-encoding>`_
    """

    def __init__(
        self,
        nda: np.ndarray | ak.Array | None = None,
        shape: tuple[int, ...] = (),
        fill_val: int | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """
        See :class:`Array`

        Parameters
        ----------
        nda
            An :class:`numpy.ndarray` or :class:`ak.Array` to be used for this
            object's internal array. If the Awkward array carries a ``units``
            parameter, it will be forwarded as LGDO attribute.
        shape
            A numpy-format shape specification for shape of the internal
            ndarray. Required if `nda` is ``None``, otherwise unused.
        fill_val
            If ``None``, memory is allocated without initialization. Otherwise,
            the array is allocated with all elements set to the corresponding
            fill value. If `nda` is not ``None``, this parameter is ignored.
        attrs
            A set of user attributes to be carried along with this LGDO. These
            attributes have always precedence over all the others (e.g. those
            carried by `nda`).
        """

        # Force uint32 behavior in superclass initialization.
        super().__init__(
            nda=nda, shape=shape, dtype=np.uint32, fill_val=fill_val, attrs=attrs
        )
        if self.dtype != np.uint32:
            msg = "ArrayOfDetectorIDs only supports dtype uint32"
            raise ValueError(msg)

    def form_datatype(self) -> str:
        dt = self.datatype_name()
        nd = str(len(self.nda.shape))
        return dt + "<" + nd + ">{detectorid}"
