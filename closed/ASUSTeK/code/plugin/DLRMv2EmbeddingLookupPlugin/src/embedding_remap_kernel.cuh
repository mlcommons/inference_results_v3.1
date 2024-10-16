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
#pragma once

#include <cstdio>

#include <cuda_pipeline_primitives.h>

#include "common_ptx.cuh"
#include "smem_staging.cuh"

template <typename Embed_t>
__global__ void remapEmbeddingRows(const Embed_t* __restrict srcEmbeddings, Embed_t* __restrict dstEmbeddings,
    const int* __restrict newLocations, const int embeddingSize, const int embeddingRows,
    const int maxEmbeddingRowsOnGpu)
{
    const int rowId = blockIdx.x;
    const int elemId = threadIdx.x;
    const int newRowPos = newLocations[rowId];

    const Embed_t val = srcEmbeddings[(long long) rowId * embeddingSize + elemId];
    if (newRowPos < maxEmbeddingRowsOnGpu)
        dstEmbeddings[(long long) newRowPos * embeddingSize + elemId] = val;
}
