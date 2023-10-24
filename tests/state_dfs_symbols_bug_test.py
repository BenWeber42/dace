import pytest

import dace
from dace.sdfg import validation


def test_state_fs_symbols_bug():
    sdfg = dace.SDFG(name="test")

    start_state = sdfg.add_state(label="start", is_start_state=True)
    left_state = sdfg.add_state(label="left")
    right_state = sdfg.add_state(label="right")

    # we define the symbol `x` from start state to the left state (with value `0`)
    sdfg.add_edge(start_state, left_state, dace.sdfg.InterstateEdge(assignments=dict(x="0")))
    # we define the symbol `y` from start state to the right state and use the symbol `x`.
    # However, `x` is only defined from start state to left state, so undefined here!
    sdfg.add_edge(start_state, right_state, dace.sdfg.InterstateEdge(assignments=dict(y="x")))

    with pytest.raises(validation.InvalidSDFGInterstateEdgeError):
        sdfg.validate()


if __name__ == '__main__':
    test_state_fs_symbols_bug()
