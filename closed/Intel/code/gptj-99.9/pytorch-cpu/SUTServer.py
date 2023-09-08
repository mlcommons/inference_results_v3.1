from argparse import ArgumentParser
import os
import sys
import time
import multiprocessing as mp
import array
import numpy as np
import threading
import subprocess
import logging
import collections

from item import InputItem, OutputItem
import thread_binder

#from dataset import Dataset
#from backend import Backend
DEBUG_PRINT=True #False

import mlperf_loadgen as lg

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SUT")
init_time = time.time()

class Consumer(mp.Process):
    def __init__(self, model_checkpoint_path="", precision="int8", quantized_model="", dataset_path="", input_queue=None, hp_queue=None, out_queue=None, lock=None, cond_var=None, init_counter=None, proc_idx=None, start_core_idx=0, cpus_per_proc=56, workers_per_proc=1, warmup=False, total_sample_count=1000, pad_inputs=False, input_lens=None, hp_threshold=1600, max_dynamic_batch_size=2, numa_offset=0):

        mp.Process.__init__(self)
        self.num_workers = workers_per_proc
        self.task_queue = input_queue
        self.hp_queue = hp_queue
        self.out_queue = out_queue
        self.lock = lock
        self.init_counter = init_counter
        self.proc_idx = proc_idx
        self.num_cores = cpus_per_proc
        self.start_core_idx = start_core_idx
        self.end_core_idx = start_core_idx + self.num_cores - 1
        self.affinity = list(range(self.start_core_idx, self.start_core_idx + self.num_cores))

        self.dataset_path = dataset_path

        self.cpus_per_worker = self.num_cores // self.num_workers
        self.workers = []
        self.out_queue = out_queue
        self.warmup = warmup
        self.latencies = collections.defaultdict(list)

        self.total_sample_count = total_sample_count
        self.pad_inputs = pad_inputs
        self.hp_threshold = hp_threshold
        self.max_dynamic_batch_size = max_dynamic_batch_size
        self.model_checkpoint_path = model_checkpoint_path
        self.precision = precision
        self.quantized_model = quantized_model
        self.cond_var = cond_var

        self.input_lens = input_lens
        self.numa_offset = numa_offset


    def doWarmup(self, wid=0):
        warmup_data = self.data_obj.getWarmupSamples()
        log.info("Starting warmup")
        input_ids, input_len, attention_mask = warmup_data[0]
        output = self.model.predict(input_ids, attention_mask)
        output = self.model.predict(input_ids, attention_mask)
        max_tokens = self.model.generate_kwargs["max_new_tokens"]
        min_tokens = self.model.generate_kwargs["min_new_tokens"]
        self.model.generate_kwargs["max_new_tokens"] = 4
        self.model.generate_kwargs["min_new_tokens"] = 4
        total = len(warmup_data)
        for i, (input_ids, input_len, attention_mask) in enumerate(warmup_data):
            if DEBUG_PRINT:
                print(f"P{self.proc_idx}-{wid}: Warm up {i}/{total} len = {input_len}")
            output = self.model.predict(input_ids, attention_mask)

        self.model.generate_kwargs["max_new_tokens"] = max_tokens
        self.model.generate_kwargs["min_new_tokens"] = min_tokens
        log.info("Process {} Warmup Completed".format(self.pid))
        with self.cond_var:
            self.init_counter.value += 1
            self.cond_var.notify()


    def handleTasks(self, i, task_queue, hp_queue, result_queue, pid, start_core, num_cores):
        thread_binder.bind_thread(start_core, num_cores)
        worker_name = str(pid) + "-" + str(i)

        # Do Warmup
        if self.warmup:
            self.doWarmup(i)

        else:
            with self.cond_var:
                self.init_counter.value += 1
                self.cond_var.notify()

        stop = False
        if DEBUG_PRINT:
            print(f"max_dynamic_batch_size = {self.max_dynamic_batch_size}, hp_threshold = {self.hp_threshold}")
        while not stop:
            try:
                tw = time.time()
                try:
                    next_task = hp_queue.get(block=False)
                except:
                    next_task = task_queue.get()
                if next_task is None:
                    log.info("Exiting worker thread : {}".format(i))
                    stop = True
                    break

                t0 = time.time()
                query_id_list = next_task.query_id_list
                sample_index_list = next_task.sample_index_list
                input_seq_lens = next_task.input_seq_lens
                label = next_task.label
                ts = next_task.receipt_time
                delay = t0 - ts
                while len(sample_index_list) < self.max_dynamic_batch_size and max(input_seq_lens) < self.hp_threshold and task_queue.qsize() > 0 and delay < 7.0:
                    try:
                        next_task = task_queue.get(block=False)
                        if next_task is not None:
                            if next_task.input_seq_lens[0] > self.hp_threshold or abs(next_task.input_seq_lens[0]-min(input_seq_lens)) > 500:
                                hp_queue.put(next_task)
                                if DEBUG_PRINT:
                                    print(f"pushing next_task = {next_task.input_seq_lens[0]}")
                                break
                            query_id_list += next_task.query_id_list
                            sample_index_list += next_task.sample_index_list
                            input_seq_lens += next_task.input_seq_lens
                            label += next_task.label
                        else:
                            stop = True
                            break
                    except:
                        break

                orig_lens = input_seq_lens
                input_ids, input_seq_lens, attention_mask = self.data_obj.getSamples(sample_index_list)

                output = self.model.predict(input_ids, attention_mask=attention_mask)
                result = self.data_obj.postProcess(query_id_list, sample_index_list, output, input_seq_lens)
                result_queue.put(result)
                t1 = time.time()
                if DEBUG_PRINT:
                    print(f"P{self.proc_idx}-{i}: time: {t1-t0:6.2f} lat: {t1-ts:6.2f} wait: {t0-tw:6.2f} delay: {delay:6.3f} input_ids = {list(input_ids.shape)} output = {list(output.shape)} out_len: {output.shape[1] - input_ids.shape[1]} ts= {ts-init_time:.2f} t0= {t0-init_time:.2f} ids = {sample_index_list} lens = {orig_lens} srno = {label}")
                task_queue.task_done()
            except Exception as ex:
                # Error occured
                log.error(ex)
                break
                self.terminate()
                sys.exit(1)


    def run(self):
        #self.proc_idx = self.pid
        os.sched_setaffinity(0, self.affinity)
        from numa import memory
        memory.set_membind_nodes(self.numa_offset+self.proc_idx) 
        print(f"P{self.proc_idx}: membind to {memory.get_membind_nodes()}")

        from backend import Backend
        self.model = Backend(model_checkpoint=self.model_checkpoint_path,
                precision=self.precision,
                quantized_model=self.quantized_model
                )

        # Load model
        log.info("Loading model")
        self.model.loadModel()
        
        from dataset import Dataset
        self.data_obj = Dataset(self.dataset_path, model_checkpoint_path=self.model_checkpoint_path, total_sample_count=self.total_sample_count, pad_inputs=self.pad_inputs)
        
        # Load Dataset
        log.info("Loading Dataset")
        self.data_obj.loadDataset()

        # Get input sequence lengths
        if self.proc_idx==0:
            with self.cond_var:
                for input_len in self.data_obj.getInputLengths():
                    self.input_lens.append(input_len)
                self.cond_var.notify()

        start_core = self.start_core_idx
        cores_left = self.num_cores

        for i in range(self.num_workers):
            log.info("Creating worker {}".format(i))
            worker_cores = min(self.cpus_per_worker, cores_left)
            cores_left -= self.cpus_per_worker
            
            worker = mp.Process(target=self.handleTasks, args=(i, self.task_queue, self.hp_queue, self.out_queue, self.pid, start_core, worker_cores))

            self.workers.append(worker)
            start_core += self.cpus_per_worker

        for w in self.workers:
            w.start()

        for w in self.workers:
            w.join()

        log.info("{} : Exiting consumer process".format(os.getpid()))


class SUT(object):
    def __init__(self, num_proc, cpus_per_proc, model_checkpoint_path, initial_core=0, batch_size=1, dataset_path=None, workers_per_proc=1, warmup=False, precision="int8", quantized_model=None, total_sample_count=1000, pad_inputs=False, hp_threshold=1600, max_dynamic_batch_size=2, numa_offset=0):

        self.num_proc = num_proc
        self.cpus_per_proc = cpus_per_proc
        self.initial_core = initial_core
        self.procs = [None] * self.num_proc
        self.workers_per_proc = workers_per_proc
        self.warmup = warmup
        self.total_workers = self.num_proc * self.workers_per_proc

        self.model_checkpoint_path = model_checkpoint_path
        self.precision = precision
        self.quantized_model = quantized_model

        self.batch_size = batch_size
        self.dataset_path = dataset_path

        self.total_sample_count = total_sample_count
        self.pad_inputs = pad_inputs
        self.hp_threshold = hp_threshold
        self.max_dynamic_batch_size = max_dynamic_batch_size
        self.numa_offset = numa_offset

        self.lock = mp.Lock()
        self.init_counter = mp.Value("i", 0)
        self.input_queue = mp.JoinableQueue()
        self.hp_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.total_samples = 0
        self.last_recv_time = None
        
        self.cv = mp.Condition(lock=self.lock)

        from multiprocessing import Manager
        
        self.input_lens = Manager().list([])

    def flushQueries(self):
        pass

    def processLatencies(self, latencies):
        pass

    def loadSamplesToRam(self, query_samples):
        pass

    def unloadSamplesFromRam(self, query_samples):
        pass

    def stopSUT(self):
        """ Stops processes and threads and exit """

        for _ in range(self.total_workers):
            self.input_queue.put(None)

        for proc in self.procs:
            proc.join()

        self.output_queue.put(None)

    def startSUT(self):
        """ Creates and Starts the processes and threads"""
        print('Server startSUT')
        # Create processes
        self.createProcesses()

        # Start processes
        log.info("Starting processes")
        for proc in self.procs:
            proc.start()
        
        # Wait for all consumers to be ready (including if they're warming up)
        with self.cv:
            #self.cv.wait_for(lambda : self.init_counter.value==self.num_proc)
            self.cv.wait_for(lambda : self.init_counter.value==self.total_workers)

        # Start Loadgen response thread
        self.response_thread = threading.Thread(target=self.responseLoadgen)
        self.response_thread.start()

    def responseLoadgen(self):
        while True:
            next_task = self.output_queue.get()
            
            if next_task is None:
                log.info('Exiting response thread')
                break

            query_id_list = next_task.query_id_list
            processed_result = next_task.result
            array_type_code = next_task.array_type_code
            batch_size = len(query_id_list)

            for id, out in zip(query_id_list, processed_result):
                response_array = array.array(array_type_code, out.tobytes())
                bi = response_array.buffer_info()
                responses = [lg.QuerySampleResponse(id, bi[0], bi[1]*response_array.itemsize)]
                lg.QuerySamplesComplete(responses)

    def createProcesses(self):
        """ Create 'mp' instances or processes"""

        start_core = self.initial_core
        for proc_idx in range(self.num_proc):
            self.procs[proc_idx] = Consumer(self.model_checkpoint_path, self.precision, self.quantized_model, self.dataset_path, self.input_queue, self.hp_queue, self.output_queue, self.lock, self.cv, self.init_counter, proc_idx, start_core, self.cpus_per_proc, self.workers_per_proc, warmup=self.warmup, total_sample_count = self.total_sample_count, pad_inputs=self.pad_inputs, input_lens=self.input_lens if proc_idx==0 else None, hp_threshold=self.hp_threshold, max_dynamic_batch_size=self.max_dynamic_batch_size, numa_offset=self.numa_offset)

            start_core += self.cpus_per_proc

    def issueQueries(self, query_samples):
        """ Receives queries and adds them to queue for processing"""
        # TODO: Implement Server logic in separate issueQuery

        num_samples = len(query_samples)
        ids = []        # Response Ids
        indexes = []    # Sample Indexes
        input_token_ids = []
        input_seq_lens = []
        label = []
        ts = time.time()
        if self.last_recv_time is None: self.last_recv_time = ts

        self.total_samples += num_samples
        if DEBUG_PRINT:
            print(f"issueQueries: num_samples {num_samples}  id: {query_samples[0].index} len: {self.input_lens[query_samples[0].index]} so far : {self.total_samples}, qsize: {self.input_queue.qsize()} ts: {ts-init_time:.2f} gap: {ts - self.last_recv_time:6.2f}")
        self.last_recv_time = ts
        if num_samples > 1:
            query_samples.sort(key=lambda x : self.input_lens[x.index])

        for i in range( num_samples):
            if len(ids)==self.batch_size:
                item = InputItem(ids, indexes, input_seq_lens=input_seq_lens, label=label, receipt_time=ts)
                if max(input_seq_lens) > self.hp_threshold and self.input_queue.qsize() > 1:
                    self.hp_queue.put(item)
                else:
                    self.input_queue.put(item)
                ids = []
                indexes = []
                input_token_ids = []
                input_seq_lens = []
                label = []

            ids.append(query_samples[i].id)
            index = query_samples[i].index
            slen = self.input_lens[index]
            indexes.append(index)
            input_seq_lens.append(slen)
            label.append(self.total_samples - num_samples + i + 1)

        if ids:
            item = InputItem(ids, indexes, input_seq_lens=input_seq_lens, label=label, receipt_time=ts)#, input_token_ids)
            if max(input_seq_lens) > self.hp_threshold and self.input_queue.qsize() > 1:
                self.hp_queue.put(item)
            else:
                self.input_queue.put(item)


