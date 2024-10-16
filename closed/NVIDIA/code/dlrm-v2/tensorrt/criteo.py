# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
from code.common.systems.system_list import SystemClassifications
if not SystemClassifications.is_soc():
    from torchrec.datasets.criteo import (
        CAT_FEATURE_COUNT,
        DAYS,
        DEFAULT_CAT_NAMES,
        DEFAULT_INT_NAMES,
    )

try:
    import mlperf_loadgen as lg
except:
    print("Loadgen Python bindings are not installed. Functionality may be limited.")

import numpy as np
import os


CRITEO_SYNTH_MULTIHOT_N_EMBED_PER_FEATURE = [40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000, 3067956,
                                             405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000, 40000000, 40000000,
                                             590152, 12973, 108, 36]
CRITEO_SYNTH_MULTIHOT_SIZES = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]


class CriteoDay23Dataset:
    """Represents the Day 23 Criteo Dataset used for MLPerf Inference.
    """

    def __init__(self, data_dir: os.PathLike, mode: str = "full"):
        self.data_dir = Path(data_dir)
        self.mode = mode

        self.labels_path = self.data_dir / "day_23_labels.npy"
        self.dense_path = self.data_dir / "day_23_dense.npy"
        self.sparse_multihot_dir = self.data_dir / "day_23_sparse_multi_hot_unpacked"

        self.labels, self.dense, self.sparse = self.load_data()

    @property
    def size(self):
        if self.mode == "full":
            return len(self.dense)
        elif self.mode == "validation":
            n_samples = len(self.dense)
            half_index = int(n_samples // 2 + n_samples % 2)
            return half_index
        else:
            raise ValueError("invalid mode")

    def sparse_input_path(self, feat_idx: int):
        return self.sparse_multihot_dir / f"{feat_idx}.npy"

    def load_data(self):
        print("Loading labels...")
        labels = np.load(self.labels_path)

        print("Loading dense inputs...")
        dense_inputs = np.load(self.dense_path)

        print("Loading sparse inputs...")
        sparse_inputs = []
        for i in range(CAT_FEATURE_COUNT):
            print(f"\tLoading Categorical feature {i}...")
            sparse_inputs.append(np.load(self.sparse_input_path(i)))

        return labels, dense_inputs, sparse_inputs

    def generate_val_map(self, val_map_dir: os.PathLike):
        """Generate sample indices for validation set. The validation set is the first half of Day 23, according to the
        reference implementation:
        https://github.com/mlcommons/inference/blob/master/recommendation/dlrm_v2/pytorch/python/multihot_criteo.py#L323
        """
        n_samples = len(self.dense)
        half_index = int(n_samples // 2 + n_samples % 2)
        val_map_txt = Path(val_map_dir) / "val_map.txt"
        with val_map_txt.open(mode='w') as f:
            for i in range(half_index):
                print(f"{i:08d}", file=f)

    def generate_cal_map(self, cal_map_dir: os.PathLike):
        """Generate sample indices for calibration set. The calibration set, according the reference implementation at
        https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch#calibration-set
        is indices 89137319 through 89265318 inclusive.
        """
        lower_bound = 89137319
        upper_bound = 89265318
        assert (upper_bound - lower_bound + 1 == 128000)
        cal_map_txt = Path(cal_map_dir) / "cal_map.txt"
        with cal_map_txt.open(mode='w') as f:
            for i in range(lower_bound, upper_bound + 1):
                print(f"{i:08d}", file=f)

    def dump_reformatted(self, dst_dir, dtype):
        new_labels_path = dst_dir / "day_23_labels.npy"
        new_dense_path = dst_dir / "day_23_dense.npy"
        new_sparse_multihot_dir = dst_dir / "day_23_sparse_multi_hot_unpacked"

        np.save(new_labels_path, np.array(self.labels, dtype=dtype))
        np.save(new_dense_path, np.array(self.dense, dtype=dtype))
        for i, arr in enumerate(self.sparse):
            full_path = new_sparse_multihot_dir / f"{i}.npy"
            np.save(full_path, np.array(arr, dtype=dtype))

    def get_batch(self, num_samples=None, indices=None):
        if indices is None:
            assert num_samples is not None
            indices = np.random.choice(self.size, size=num_samples, replace=False)
        batch = {
            "dense": self.dense[indices],
            "sparse": [self.sparse[i][indices] for i in range(CAT_FEATURE_COUNT)],
            "labels": self.labels[indices],
        }
        return batch

    def dump_concatenated_sparse_input(self):
        concatenated = np.hstack(self.sparse)
        assert concatenated.shape == (178274637, sum(CRITEO_SYNTH_MULTIHOT_SIZES))

        concat_path = self.data_dir / "day_23_sparse_concatenated.npy"
        np.save(concat_path, concatenated)


def convert_sample_partition_to_npy(txt_path: os.PathLike):
    p = Path(txt_path)
    # Need to convert to a numpy file
    indices = [0]
    with p.open() as f:
        while (line := f.readline()):
            if len(line) == 0:
                continue

            start, end, count = line.split(", ")
            assert int(start) == indices[-1]
            indices.append(int(end))
    partition = np.array(indices, dtype=np.int32)
    np.save(p.with_suffix(".npy"), partition)
    return partition


class CriteoQSL:
    def __init__(self,
                 ds: CriteoDay23Dataset,
                 partition_path: os.PathLike = "/home/mlperf_inf_dlrmv2/criteo/day23/sample_partition.txt"):
        self.ds = ds
        self.partitions = self.load_partition(Path(partition_path))
        self.item_count = len(self.partitions) - 1
        self.active_ids = dict()

    def load_partition(self, p):
        partition = None
        if p.suffix == ".txt":
            partition = convert_sample_partition_to_npy(p)
        elif p.suffix == ".npy":
            partition = np.load(p)
        else:
            raise RuntimeError("Sample partition file must be a .txt or .npy file")
        return partition

    def unload_query_samples(self, sample_list):
        for sample_idx in sample_list:
            self.active_ids.pop(sample_idx)

    def load_query_samples(self, sample_list):
        # Criteo is weird, we need to translate sample indices to true dataset indices using the sample partition
        for sample_idx in sample_list:
            self.active_ids[sample_idx] = np.arange(self.partitions[sample_idx],
                                                    self.partitions[sample_idx + 1],
                                                    dtype=np.int32)

    def get_query_samples(self, sample_list):
        return {
            idx: self.active_ids[idx]
            for idx in sample_list
        }

    def as_loadgen_qsl(self, total_sample_count, performance_sample_count):
        return lg.ConstructQSL(total_sample_count,
                               performance_sample_count,
                               self.load_query_samples,
                               self.unload_query_samples)
