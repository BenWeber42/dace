# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import math
import copy
import numpy as np

from typing import Generator, Dict, List, Tuple
from collections import Counter

from dace import SDFG, dtypes
from dace.optimization import cutout_tuner
from dace.sdfg.analysis import cutout as cutter

from dace.transformation import subgraph as sg
from dace.transformation.estimator import enumeration as en
from dace.transformation.subgraph import helpers
from dace.optimization import utils as optim_utils

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x, **kwargs: x


class OnTheFlyMapFusionTuner(cutout_tuner.CutoutTuner):

    def __init__(self, sdfg: SDFG, measurement: dtypes.InstrumentationType = dtypes.InstrumentationType.Timer) -> None:
        super().__init__(task="OnTheFlyMapFusion", sdfg=sdfg)
        self.instrument = measurement

    def cutouts(self):
        for state in self._sdfg.nodes():
            state_id = self._sdfg.node_id(state)
            nodes = state.nodes()

            try:
                cutout = cutter.cutout_state(state, *(nodes), make_copy=False)
                yield cutout, f"{state_id}.{state.label}"
            except AttributeError:
                continue

    def config_from_key(self, key: str, cutout: dace.SDFG, **kwargs) -> Tuple[int, List[int]]:
        fusion_id = int(key)
        if fusion_id == 0:
            return (0, [])

        sp = list(self.space(cutout=cutout))
        return sp[fusion_id]

    def space(self, cutout: dace.SDFG) -> Generator[List[bool], None, None]:
        subgraphs = en.ConnectedEnumerator(cutout, cutout.start_state)
        yield 0, []

        for i, (subgraph, score) in enumerate(subgraphs):
            yield i + 1, list(map(lambda m: cutout.start_state.node_id(m), subgraph))

    def pre_evaluate(self, cutout: dace.SDFG, measurements: int, **kwargs) -> Dict:
        cutout.start_state.instrument = self.instrument

        new_kwargs = {
            "space_kwargs": {
                "cutout": cutout
            },
            "cutout": cutout.to_json(),
            "measurements": measurements,
            "key": lambda point: str(point[0])
        }
        return new_kwargs

    def evaluate(self, config, cutout, measurements: int, **kwargs) -> float:
        candidate = dace.SDFG.from_json(cutout)
        for node in candidate.start_state:
            if isinstance(node, dace.nodes.MapEntry):
                break
        else:
            # Skip no-map-states
            return math.inf

        if config[0] == 0:
            # Baseline
            return self.measure(candidate, measurements)

        map_ids = config[1]
        if len(map_ids) < 2:
            return math.inf

        maps_ = list(map(candidate.start_state.node, map_ids))
        subgraph = helpers.subgraph_from_maps(sdfg=candidate, graph=candidate.start_state, map_entries=maps_)

        map_fusion = sg.MapFusion(subgraph, candidate.sdfg_id, candidate.node_id(candidate.start_state))
        if map_fusion.can_be_applied(candidate.start_state, candidate):
            fuse_counter = map_fusion.apply(candidate.start_state, candidate)

            if fuse_counter == 0:
                return math.inf

        return self.measure(candidate,  measurements)

    def apply(self, config: Tuple[int, List[int]], label: str, **kwargs) -> None:
        if config[0] == 0:
            return

        state_id = label.split(".")[0]
        state_id = int(state_id)
        state = self._sdfg.node(state_id)
        nodes = state.nodes()
        cutout = cutter.cutout_state(state, *(nodes), make_copy=False)

        map_ids = config[1]
        maps_ = list(map(cutout.start_state.node, map_ids))
        subgraph = helpers.subgraph_from_maps(sdfg=self._sdfg, graph=state, map_entries=maps_)

        map_fusion = sg.MapFusion(subgraph, self._sdfg.sdfg_id, state_id)
        if map_fusion.can_be_applied(state, self._sdfg):
            fuse_counter = map_fusion.apply(state, self._sdfg)
            print(f"Fusing {fuse_counter} maps")

    def _extract_patterns(self, configs: List[Tuple[str, List[int]]]):
        # Describe successful fusions as set of map descriptors
        subgraph_patterns = []
        for label, config in configs:
            state_id = label.split(".")[0]
            state_id = int(state_id)
            state = self._sdfg.node(state_id)
            nodes = state.nodes()
            cutout = cutter.cutout_state(state, *(nodes), make_copy=False)

            pattern_desc = Counter()
            fusion_id, map_ids = self.config_from_key(config, cutout)
            if fusion_id == 0:
                continue

            for map_id in map_ids:
                map_entry = cutout.start_state.node(map_id)
                map_desc = OnTheFlyMapFusionTuner.map_descriptor(cutout.start_state, map_entry)
                pattern_desc.update({map_desc: 1})

            subgraph_patterns.append(pattern_desc)

        return subgraph_patterns

    @staticmethod
    def transfer(sdfg: dace.SDFG, tuner, k: int = 5):
        assert isinstance(tuner, OnTheFlyMapFusionTuner)

        dreport = sdfg.get_instrumented_data()
        assert dreport is not None

        tuning_report = tuner.optimize(apply=False)
        best_configs = cutout_tuner.CutoutTuner.top_k_configs(tuning_report, k=k)
        subgraph_patterns = tuner._extract_patterns(best_configs)
        for state in list(sdfg.nodes()):
            top_maps = helpers.get_outermost_scope_maps(sdfg, state)
            if len(top_maps) < 2:
                print("Skipping: ", state.label)
                continue
                
            try:
                cutout = cutter.cutout_state(state, *(state.nodes()), make_copy=False)
                cutout.start_state.instrument = dace.InstrumentationType.GPU_Events
            except AttributeError as e:
                print(e)
                print("Skipping: ", state.label)
                continue
                
            initial_runtime = optim_utils.measure(cutout, dreport)
            if initial_runtime == math.inf:
                print("Skipping: ", state.label)
                continue

            # Try to apply every subgraph_pattern greedily, i.e., highest expected speedup first
            for pattern in subgraph_patterns:
                maps = helpers.get_outermost_scope_maps(sdfg, state)
                if len(maps) < 2:
                    continue

                maps_desc = {}
                state_desc = Counter()
                for map_entry in maps:
                    map_desc = OnTheFlyMapFusionTuner.map_descriptor(state, map_entry)
                    state_desc.update({map_desc: 1})
                    
                    if not map_desc in maps_desc:
                        maps_desc[map_desc] = []

                    maps_desc[map_desc].append(map_entry)
                
                included = True
                for key in pattern:
                    if not key in state_desc or pattern[key] > state_desc[key]:
                        included = False                        
                        break

                if not included:
                    continue

                # Construct subgraph greedily
                subgraph_maps = []
                for desc in pattern:
                    num = pattern[desc]
                    subgraph_maps.extend(maps_desc[desc][:num])

                # Apply
                experiment_sdfg = copy.deepcopy(sdfg)
                experiment_state = experiment_sdfg.node(sdfg.node_id(state))
                
                experiment_subgraph_maps = list(map(lambda me: experiment_state.node(state.node_id(me)), subgraph_maps))
                experiment_subgraph = helpers.subgraph_from_maps(sdfg=experiment_sdfg, graph=experiment_state, map_entries=experiment_subgraph_maps)
                
                map_fusion = sg.MapFusion(experiment_subgraph, experiment_sdfg.sdfg_id, experiment_sdfg.node_id(experiment_state))
                if map_fusion.can_be_applied(experiment_state, experiment_sdfg):
                    print("Comparing")
                    baseline = cutter.cutout_state(experiment_state, *(experiment_state.nodes()), make_copy=True)                    
                    baseline.start_state.instrument = dace.InstrumentationType.GPU_Events
                    base_runtime = optim_utils.measure(baseline, dreport)

                    experiment_fuse_counter = map_fusion.apply(experiment_state, experiment_sdfg)
                    if experiment_fuse_counter == 0:
                        continue

                    experiment_cutout = cutter.cutout_state(experiment_state, *(experiment_state.nodes()), make_copy=False)
                    experiment_cutout.start_state.instrument = dace.InstrumentationType.GPU_Events
                    fused_runtime = optim_utils.measure(experiment_cutout, dreport)
                    print(base_runtime, fused_runtime, experiment_fuse_counter)
                    if fused_runtime > base_runtime:
                        continue

                    print(f"Fusing {experiment_fuse_counter} maps. Performance improvement: {base_runtime - fused_runtime}")

                    subgraph = helpers.subgraph_from_maps(sdfg=sdfg, graph=state, map_entries=subgraph_maps)
                    map_fusion = sg.MapFusion(subgraph, sdfg.sdfg_id, sdfg.node_id(state))
                    actual_fuse_counter = map_fusion.apply(state, sdfg)
                    assert actual_fuse_counter == experiment_fuse_counter

            tuned_top_maps = helpers.get_outermost_scope_maps(sdfg, state)
            print(len(top_maps), len(tuned_top_maps))

            cutout_tuned = cutter.cutout_state(state, *(state.nodes()), make_copy=False)
            cutout_tuned.start_state.instrument = dace.InstrumentationType.GPU_Events
            tuned_runtime = optim_utils.measure(cutout_tuned, dreport)
            
            print(f"Tuning result of {state.label} (initial, tuned): {initial_runtime, tuned_runtime}")
            print()
            print()

    @staticmethod
    def map_descriptor(state: dace.SDFGState, map_entry: dace.nodes.MapEntry) -> str:
        tasklets = filter(lambda node: isinstance(node, dace.nodes.Tasklet), map(lambda edge: edge.dst, state.out_edges(map_entry)))
        tasklets = set(tasklets)

        desc = []
        for tasklet in tasklets:
            label = tasklet.label.split("_")[:-2]
            label = "_".join(label)
            desc.append(label)

        return ":".join(desc)
