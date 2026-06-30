from __future__ import annotations

import pickle

import numpy as np

import lgdo
from lgdo import WaveformTable


def test_init():
    wft = WaveformTable()  # defaults: size = 1024 and wf_len = 100
    assert (wft.t0.nda == np.zeros(1024)).all()
    assert (wft.dt.nda == np.full(1024, fill_value=1)).all()
    assert isinstance(wft.values, lgdo.VectorOfVectors)
    assert len(wft.values) == 1024
    assert wft.attrs == {"datatype": "table{dt,t0,values}"}

    wft = WaveformTable(dt_units="ns", values_units="adc")
    assert wft.dt.attrs["units"] == "ns"
    assert wft.values.attrs["units"] == "adc"

    wft = WaveformTable(size=10, wf_len=1000)
    assert (wft.t0.nda == np.zeros(10)).all()
    assert (wft.dt.nda == np.full(10, fill_value=1)).all()
    assert isinstance(wft.values, lgdo.ArrayOfEqualSizedArrays)
    assert (wft.values.nda == np.zeros(shape=(10, 1000))).all()
    assert wft.values.nda.dtype == np.float64

    # test __getitem__
    wf_slice = wft[1]
    assert isinstance(wf_slice, WaveformTable)
    assert len(wf_slice) == 1
    assert wf_slice.values.shape == (1, 1000)
    assert (wf_slice.dt.nda == np.full(1, fill_value=1)).all()
    assert (wf_slice.t0.nda == np.zeros(1)).all()

    wf_slice = wft[1:3]
    assert isinstance(wf_slice, WaveformTable)
    assert len(wf_slice) == 2
    assert wf_slice.values.shape == (2, 1000)
    assert (wf_slice.dt.nda == np.full(2, fill_value=1)).all()
    assert (wf_slice.t0.nda == np.zeros(2)).all()

    wf_slice = wft[[1, 3, 5]]
    assert isinstance(wf_slice, WaveformTable)
    assert len(wf_slice) == 3
    assert wf_slice.values.shape == (3, 1000)
    assert (wf_slice.dt.nda == np.full(3, fill_value=1)).all()
    assert (wf_slice.t0.nda == np.zeros(3)).all()

    wft = WaveformTable(
        size=10, dt=np.zeros(5), t0=np.zeros(5), values=np.zeros((5, 50))
    )
    assert (wft.t0.nda == np.zeros(10)).all()
    assert (wft.dt.nda == np.zeros(10)).all()
    assert isinstance(wft.values, lgdo.ArrayOfEqualSizedArrays)
    assert (wft.values.nda == np.zeros(shape=(10, 50))).all()
    assert wft.values.nda.dtype == np.float64

    wft = WaveformTable(
        values=lgdo.ArrayOfEqualSizedArrays(shape=(10, 1000), fill_val=69)
    )
    assert (wft.t0.nda == np.zeros(10)).all()
    assert (wft.dt.nda == np.full(10, fill_value=1)).all()
    assert isinstance(wft.values, lgdo.ArrayOfEqualSizedArrays)
    assert (wft.values.nda == np.full(shape=(10, 1000), fill_value=69)).all()

    wft = WaveformTable(values=np.full(shape=(10, 1000), fill_value=69))
    assert (wft.t0.nda == np.zeros(10)).all()
    assert (wft.dt.nda == np.full(10, fill_value=1)).all()
    assert isinstance(wft.values, lgdo.ArrayOfEqualSizedArrays)
    assert (wft.values.nda == np.full(shape=(10, 1000), fill_value=69)).all()

    wft = WaveformTable(
        values=lgdo.VectorOfVectors(shape_guess=(10, 1000), dtype=np.float32)
    )
    assert (wft.t0.nda == np.zeros(10)).all()
    assert (wft.dt.nda == np.full(10, fill_value=1)).all()
    assert isinstance(wft.values, lgdo.VectorOfVectors)
    assert len(wft.values) == 10

    wft = WaveformTable(
        values=lgdo.VectorOfEncodedVectors(
            encoded_data=lgdo.VectorOfVectors(shape_guess=(10, 1000), dtype=np.ubyte),
            decoded_size=lgdo.Array(shape=10, fill_val=6),
        )
    )
    assert (wft.t0.nda == np.zeros(10)).all()
    assert (wft.dt.nda == np.full(10, fill_value=1)).all()
    assert isinstance(wft.values, lgdo.VectorOfEncodedVectors)
    assert len(wft.values) == 10

    wft = WaveformTable(
        t0=[1, 1, 1], dt=[2, 2, 2], values=lgdo.ArrayOfEqualSizedArrays(shape=(3, 1000))
    )
    assert (wft.t0.nda == np.full(3, fill_value=1)).all()
    assert (wft.dt.nda == np.full(3, fill_value=2)).all()
    assert isinstance(wft.values, lgdo.ArrayOfEqualSizedArrays)
    assert wft.values.nda.shape == (3, 1000)

    wft = WaveformTable(
        values=lgdo.ArrayOfEncodedEqualSizedArrays(
            encoded_data=lgdo.VectorOfVectors(shape_guess=(10, 1000), dtype=np.ubyte),
            decoded_size=10,
        )
    )
    assert (wft.t0.nda == np.zeros(10)).all()
    assert (wft.dt.nda == np.full(10, fill_value=1)).all()
    assert isinstance(wft.values, lgdo.ArrayOfEncodedEqualSizedArrays)
    assert len(wft.values) == 10

    wft = WaveformTable(t0=[1, 1, 1], dt=[2, 2, 2], wf_len=1000)
    assert (wft.t0.nda == np.full(3, fill_value=1)).all()
    assert (wft.dt.nda == np.full(3, fill_value=2)).all()
    assert isinstance(wft.values, lgdo.ArrayOfEqualSizedArrays)
    assert wft.values.nda.shape == (3, 1000)

    wft = WaveformTable(t0=[1, 1, 1], dt=[2, 2, 2], wf_len=1000, dtype=np.float32)
    assert wft.values.nda.dtype == np.float32

    wft = WaveformTable(10, wf_len=20)
    wft.wf_len = 30
    assert wft.wf_len == 30


def test_pickle():
    wft = WaveformTable(size=10, wf_len=1000)
    wft_pickled = pickle.dumps(wft)
    wft_unpickled = pickle.loads(wft_pickled)
    assert (wft.t0.nda == wft_unpickled.t0.nda).all()
    assert (wft.dt.nda == wft_unpickled.dt.nda).all()
    assert (wft.values.nda == wft_unpickled.values.nda).all()
