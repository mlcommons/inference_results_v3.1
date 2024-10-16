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

#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>

#include "dlrmv2Helper.h"
#include "embedding_remap_kernel.cuh"
#include "embedding_remap_launch.h"

void remapEmbeddingRows(cudaStream_t stream, const void* srcEmbeddings, void* dstEmbeddings, const int* newLocations,
    const int embeddingSize, const int embeddingRows, const int maxEmbeddingRowsOnGpu, const int embed_elem_size)
{
    if (embed_elem_size == sizeof(float))
        remapEmbeddingRows<float><<<embeddingRows, embeddingSize, 0, stream>>>(static_cast<const float*>(srcEmbeddings),
            static_cast<float*>(dstEmbeddings), newLocations, embeddingSize, embeddingRows, maxEmbeddingRowsOnGpu);

    else if (embed_elem_size == sizeof(half))
        remapEmbeddingRows<half><<<embeddingRows, embeddingSize, 0, stream>>>(static_cast<const half*>(srcEmbeddings),
            static_cast<half*>(dstEmbeddings), newLocations, embeddingSize, embeddingRows, maxEmbeddingRowsOnGpu);

    CUDA_ASSERT(cudaGetLastError());
}
