/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdio>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void select_top_kernel(const void* input, void* output, int32_t bs, int32_t vocab_size)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < bs)
    {
        const half* row_ptr = static_cast<const half*>(input) + tid * vocab_size;
        half max_val = -INFINITY;
        int max_idx = 0;
        for (int i = 0; i < vocab_size; i++)
        {
            const half val = row_ptr[i];
            if (val > max_val)
            {
                max_val = val;
                max_idx = i;
            }
        }
        int32_t* output_32 = static_cast<int32_t*>(output);
        output_32[tid] = max_idx;
    }
}

void launch_top_kernel(const void* input, void* output, int32_t bs, int32_t vocab_size, cudaStream_t stream)
{
    select_top_kernel<<<1, bs, 0, stream>>>(input, output, bs, vocab_size);
}
