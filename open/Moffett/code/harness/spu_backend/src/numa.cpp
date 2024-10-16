/*
 * Copyright © 2023 Moffett System Inc. All rights reserved.
 *
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

#include "numa.h"

#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <linux/ioctl.h>
#include <mutex>

namespace moffett {
namespace spu_backend {

#define MOFFETT_MMAP_NODE "/dev/mf-remap-pfn"

#define MOFFETT_MMAP_IOC_MAGIC  	'k'
#define MOFFETT_MMAP_IOCMALLOC        _IOWR(MOFFETT_MMAP_IOC_MAGIC, 0x1, unsigned long)
#define MOFFETT_MMAP_IOCFREE          _IOWR(MOFFETT_MMAP_IOC_MAGIC, 0x2, unsigned long)
#define MOFFETT_MMAP_IOCFREE_ALL      _IOWR(MOFFETT_MMAP_IOC_MAGIC, 0x3, unsigned long)
#define MOFFETT_MMAP_IOCGET_PID       _IOWR(MOFFETT_MMAP_IOC_MAGIC, 0x4, unsigned long)
#define MOFFETT_MMAP_IOCSET_NODE_ID   _IOWR(MOFFETT_MMAP_IOC_MAGIC, 0x5, unsigned long)

#define MOFFETT_MMAP_IOC_MAXNR  5

typedef struct moffett_mmap_args
{
  uint64_t vaddr;
  uint32_t pid; //used to distinguish different processes
  uint32_t node_id;
} moffett_mmap_args_t;

typedef struct _moffett_mmap_info
{
  uint32_t pid; //used to distinguish different processes
  int node_id; // numa node id
} moffett_mmap_info_t;

static moffett_mmap_info_t moffett_mmap_info = {0, -1};
static std::mutex mutex;

/**
 *@return: node id: current numa node id
 */
int moffett_mmap_get_numa_node_id(void)
{
  std::lock_guard<std::mutex> guard(mutex);
  return moffett_mmap_info.node_id;
}

/**
 **set numa node id, which used to malloc memory on this node
 *@return: 0:success; -1:fail
 */
int moffett_mmap_set_numa_node_id(int node_id)
{
  if (node_id == moffett_mmap_get_numa_node_id()) {
    return 0;
  }

  int fd;
  moffett_mmap_args_t mmap_args;

#ifdef MFT_MMAP_DEBUG
  printf("%s \n", __func__);
#endif

  fd = open(MOFFETT_MMAP_NODE, O_RDWR);
  if (fd < 0) {
    perror("/dev/mf-remap-pfn open failed \n");
    return -1;
  }

  memset(&mmap_args, 0, sizeof(moffett_mmap_args_t));
  mmap_args.node_id = node_id;
  if (ioctl(fd, MOFFETT_MMAP_IOCSET_NODE_ID, &mmap_args) < 0) {
    printf("%s, Call cmd MOFFETT_MMAP_IOCSET_NODE_ID fail! \n", __func__);
    close(fd);
    return -1;
  }

  close(fd);

  std::lock_guard<std::mutex> guard(mutex);
  moffett_mmap_info.node_id = node_id;

  return 0;
}

void* moffett_mmap(size_t size)
{
  int fd;
  void *pvaddr = NULL;

  fd = open(MOFFETT_MMAP_NODE, O_RDWR);
  if (fd < 0) {
    return NULL;
  }

  pvaddr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_LOCKED, fd, 0);
//  printf("mmap: %lu, %p, errno=%d, %s\n", (uint64_t)pvaddr, pvaddr, errno, strerror(errno));
  if (pvaddr == MAP_FAILED) {
    // handle different error code
    if (errno == EAGAIN) {
      printf("Error: Insufficient permissions in `mmap()`\n");
    } else if (errno == ENOMEM) {
      printf("Error: Out of memory in `mmap()`\n");
    }
    close(fd);
    return NULL;
  }

  close(fd);

  return pvaddr;
}

int moffett_munmap(void *pvaddr, size_t size)
{
  int ret = 0;
  int fd;
  moffett_mmap_args_t mmap_args;

#ifdef MFT_MMAP_DEBUG
  printf("%s, pvaddr:%p, size:%ld \n", __func__, pvaddr, size);
#endif

  if(!pvaddr || !size)
    return -1;

  fd = open(MOFFETT_MMAP_NODE, O_RDWR);
  if (fd < 0) {
    perror("/dev/mf-remap_pfn open failed \n");
    return -1;
  }

  munmap(pvaddr, size);

  memset(&mmap_args, 0, sizeof(moffett_mmap_args_t));
  mmap_args.vaddr = (uint64_t)pvaddr;
  if (ioctl(fd, MOFFETT_MMAP_IOCFREE, &mmap_args) < 0) {
    ret = -1;
    printf("%s, Call cmd DMABUF_IOCFREE fail! \n", __func__);
  }

  close(fd);

  return ret;
}

int moffett_munmap_all(void)
{
  int ret = 0;
  int fd;
  moffett_mmap_args_t mmap_args;

#ifdef MFT_MMAP_DEBUG
  printf("%s \n", __func__);
#endif

  fd = open(MOFFETT_MMAP_NODE, O_RDWR);
  if (fd < 0) {
    perror("/dev/mf-remap-pfn open failed \n");
    return -1;
  }

  memset(&mmap_args, 0, sizeof(moffett_mmap_args_t));
  if (ioctl(fd, MOFFETT_MMAP_IOCFREE_ALL, &mmap_args) < 0) {
    ret = -1;
    printf("%s, Call cmd MOFFETT_MMAP_IOCFREE_ALL fail! \n", __func__);
  }

  close(fd);

  return ret;
}


void *MallocHostMemory(size_t size, int numa) {
  moffett_mmap_set_numa_node_id(numa);
  return moffett_mmap(size);
}

int FreeHostMemory(void* vaddr, size_t size, int numa) {
  moffett_mmap_set_numa_node_id(numa);
  return moffett_munmap(vaddr, size);
}

int FreeHostMemoryAll() {
  return moffett_munmap_all();
}

}
}