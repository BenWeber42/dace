import dace
from dace import dtypes, nodes
import numpy as np


N = dace.symbol("N")


@dace.program
def cpu_singleton_program(A: dace.int32[N]):
    for i in dace.map[0:N] @ dtypes.ScheduleType.CPU_Multicore_Singleton:
        with dace.tasklet:
            # Assuming OpenMP is used
            t = omp_get_thread_num()
            t >> A[i]


@dace.program
def cpu_singleton_program_invalid_map_ranges(A: dace.int32[N]):
    # We can only have one range with the `CPU_Multicore_Singleton` schedule!
    for i, j in dace.map[0:N, 0:N] @ dtypes.ScheduleType.CPU_Multicore_Singleton:
        A[i] = i


def test_cpu_singleton() -> None:
    N = 128

    A = np.ndarray([N], dtype=np.int32)
    A[:] = -1

    sdfg = cpu_singleton_program.to_sdfg()

    sdfg(A, N=N)
    threads = np.max(A) + 1
    tasks = np.count_nonzero(A != -1)
    assert threads == tasks
    print(f"OK. Detected threads: {threads}")
    print(f"OK. Scheduled tasks: {tasks}/{N}")


def test_cpu_singleton_invalid_map_ranges() -> None:
    N = 128

    A = np.ndarray([N], dtype=np.int32)
    A[:] = -1

    sdfg = cpu_singleton_program_invalid_map_ranges.to_sdfg()

    try:
        sdfg(A, N=N)
    except ValueError as v:
        assert (
            "A CPU Thread Singleton map can only have one range" in v.args[0]
        ), "Wrong exception message!"
        print("OK. Exception correctly thrown.")
    except:
        assert False, "Wrong exception was thrown!"
    else:
        assert False, "No exception was throwng!"


if __name__ == "__main__":
    test_cpu_singleton()
    test_cpu_singleton_invalid_map_ranges()
