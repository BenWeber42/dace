# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import pytest

import dace
from dace import subsets


def test_memlet_subset_validation_bug():
    sdfg = dace.SDFG("test")

    sdfg.add_array("A", shape=(5, ), dtype=dace.int32)
    sdfg.add_array("B", shape=(5, ), dtype=dace.int32)
    sdfg.add_array("indirection", shape=(5, ), dtype=dace.int32)

    state = sdfg.add_state("start", is_start_state=True)
    access_A = state.add_access("A")
    access_B = state.add_access("B")
    memlet = state.add_edge(access_A, "A", access_B, None, dace.Memlet("A[0]"))

    # This is invalid SDFG as we are not allowed to use other arrays as subset expressions!
    memlet.data.subset = subsets.Range([("indirection[0]", "indirection[0]", "1")])

    # FIXME: we don't seem to have an appropriate type for memlet errors?
    with pytest.raises(Exception):
        sdfg.validate()

    # This must also work if the array is doesn't exist
    memlet.data.subset = subsets.Range([("undeclared_array[0]", "undeclared_array[0]", "1")])

    # FIXME: we don't seem to have an appropriate type for memlet errors?
    with pytest.raises(Exception):
        sdfg.validate()


if __name__ == "__main__":
    test_memlet_subset_validation_bug()
