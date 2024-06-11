# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np


def test_simple_tasklet_connectors():

    @dace.program
    def add_tasklet(A: dace.float64[1], B: dace.float64[1], R: dace.float64[1]):

        with dace.tasklet(dace.Language.Python):
            a << A[0]
            b << B[0]
            r >> R[0]
            r = a + b

    rng = np.random.default_rng()
    a = rng.random(1)
    b = rng.random(1)
    r = np.empty(1)

    add_tasklet(A=a, B=b, R=r)
    assert np.allclose(r, a + b)


if __name__ == "__main__":
    test_simple_tasklet_connectors()
