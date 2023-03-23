import dace
from dace import dtypes, nodes
import numpy as np


N = dace.symbol("N")


@dace.program
def cpu_singleton_program(A: dace.int32[N]):
    for i in dace.map[0:N]:
        A[i] = i


@dace.program
def cpu_singleton_program_invalid_map_ranges(A: dace.int32[N]):
    # We can only have one range with the `CPU_Multicore_Singleton` schedule!
    for i, j in dace.map[0:N, 0:N]:
        A[i] = i


def first_map_entry_to_cpu_singleton(sdfg: dace.SDFG) -> None:
    map_entry = None
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            map_entry = node
            break

    map_entry.schedule = dtypes.ScheduleType.CPU_Multicore_Singleton


def test_cpu_singleton() -> None:
    N = 128

    A = np.ndarray([N], dtype=np.int32)
    A[:] = -1

    sdfg = cpu_singleton_program.to_sdfg()
    first_map_entry_to_cpu_singleton(sdfg)

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
    first_map_entry_to_cpu_singleton(sdfg)

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
