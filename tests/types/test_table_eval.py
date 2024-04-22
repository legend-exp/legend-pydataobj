from __future__ import annotations

import numpy as np

import lgdo


def test_eval_dependency():
    obj = lgdo.Table(
        col_dict={
            "a": lgdo.Array([1, 2, 3, 4], attrs={"units": "s"}),
            "b": lgdo.Array([5, 6, 7, 8]),
            "c": lgdo.ArrayOfEqualSizedArrays(
                nda=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
            ),
            "d": lgdo.ArrayOfEqualSizedArrays(
                nda=[
                    [21, 22, 23, 24],
                    [25, 26, 27, 8],
                    [29, 210, 211, 212],
                    [213, 214, 215, 216],
                ],
            ),
            "e": lgdo.VectorOfVectors([[1, 2, 3], [4], [], [8, 6]]),
            "ee": lgdo.VectorOfVectors([[[1], [2, 3]], [[], [4]], [[]], [[8, 6]]]),
            "tbl": lgdo.Table(
                col_dict={
                    "z": lgdo.Array([1, 1, 1, 1]),
                    "y": lgdo.Array([8, 8, 8, 8]),
                }
            ),
        }
    )
    r = obj.eval("sum(a)")
    assert isinstance(r, lgdo.Scalar)

    r = obj.eval("a + b")
    assert isinstance(r, lgdo.Array)
    assert np.array_equal(r.nda, obj.a.nda + obj.b.nda)

    r = obj.eval("a + tbl__z")
    assert isinstance(r, lgdo.Array)
    assert np.array_equal(r.nda, obj.a.nda + obj.tbl.z.nda)

    r = obj.eval("((a - b) > 1) & ((b - a) < -1)")
    assert isinstance(r, lgdo.Array)
    assert r.dtype == "bool"

    r = obj.eval("a + d")
    assert isinstance(r, lgdo.ArrayOfEqualSizedArrays)

    assert obj.eval("a**2")
    assert obj.eval("sin(a)")
    assert obj.eval("log(d)")

    r = obj.eval("a + e")
    assert isinstance(r, lgdo.VectorOfVectors)
    assert r == lgdo.VectorOfVectors([[2, 3, 4], [6], [], [12, 10]])

    r = obj.eval("2*e + 1")
    assert isinstance(r, lgdo.VectorOfVectors)
    assert r == lgdo.VectorOfVectors([[3, 5, 7], [9], [], [17, 13]])

    r = obj.eval("2*ee + 1")
    assert isinstance(r, lgdo.VectorOfVectors)
    assert r == lgdo.VectorOfVectors([[[3], [5, 7]], [[], [9]], [[]], [[17, 13]]])

    r = obj.eval("e > 2")
    assert isinstance(r, lgdo.VectorOfVectors)
    assert r == lgdo.VectorOfVectors([[False, False, True], [True], [], [True, True]])
    assert r.dtype == "bool"

    r = obj.eval("ak.sum(e, axis=-1)")
    assert isinstance(r, lgdo.Array)

    r = obj.eval("ak.sum(e)")
    assert isinstance(r, lgdo.Scalar)

    assert obj.eval("np.sum(a) + ak.sum(e)")
