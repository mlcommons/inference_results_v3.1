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
import torch

from item import InputItem, OutputItem
import thread_binder

#from dataset import Dataset
#from backend import Backend

import mlperf_loadgen as lg

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SUT")

class Consumer(mp.Process):
    def __init__(self, library_path="",model_path="", precision="int8", quantized_model="", dataset_path="", input_queue=None, out_queue=None, lock=None, cond_var=None, init_counter=None, proc_idx=None, start_core_idx=0, cpus_per_proc=56, workers_per_proc=1, warmup=False, total_sample_count=1000, pad_inputs=False):

        mp.Process.__init__(self)
        self.num_workers = workers_per_proc
        self.library_path = library_path
        self.model_path = model_path
        self.task_queue = input_queue
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

        self.precision = precision
        self.quantized_model = quantized_model
        self.cond_var = cond_var


    def doWarmup(self):
        warmup_data = self.data_obj.getWarmupSamples()
        log.info("Starting warmup")

        for i, (input_ids, input_len, attention_mask) in enumerate(warmup_data):
            log.info("{} iter done".format(i))
            output = self.model.predict(input_ids, attention_mask)

        log.info("Process {} Warmup Completed".format(self.proc_idx))
        with self.cond_var:
            self.init_counter.value += 1
            self.cond_var.notify()


    def handleTasks(self, i, task_queue, result_queue, pid, start_core, num_cores):
        thread_binder.bind_thread(start_core, num_cores)
        worker_name = str(pid) + "-" + str(i)

        # Do Warmup
        if self.warmup:
            self.doWarmup()

        else:
            with self.cond_var:
                self.init_counter.value += 1
                self.cond_var.notify()

        while True:
            try:
                next_task = task_queue.get()
                if next_task is None:
                    log.info("Exiting worker thread : {}".format(i))
                    break

                query_id_list = next_task.query_id_list
                sample_index_list = next_task.sample_index_list
                input_seq_lens = next_task.input_seq_lens

                input_ids, input_seq_lens, attention_mask = self.data_obj.getSamples(sample_index_list)
                output = self.model.predict(input_ids, attention_mask=attention_mask)
                log.info("Done")
                output=output.unsqueeze(0)
                output=torch.cat((input_ids,output),1)
                result = self.data_obj.postProcess(query_id_list, sample_index_list, output, input_seq_lens)
                result_queue.put(result)
                task_queue.task_done()
            except Exception as ex:
                # Error occured
                log.error(ex)
                break
                self.terminate()
                sys.exit(1)

    def run(self):
        os.sched_setaffinity(0, self.affinity)

        from backend import Backend
        self.model = Backend(library_path=self.library_path,
                model_path=self.model_path,
                proc_idx=self.proc_idx
                )

        # Load model
        log.info("Loading model")
        self.model.loadModel()
        
        from dataset import Dataset
        self.data_obj = Dataset(self.dataset_path, model_checkpoint_path="EleutherAI/gpt-j-6B", total_sample_count=self.total_sample_count, pad_inputs=self.pad_inputs)       
 
        # Load Dataset
        log.info("Loading Dataset")
        self.data_obj.loadDataset()

        start_core = self.start_core_idx
        cores_left = self.num_cores

        for i in range(self.num_workers):
            log.info("Creating worker {}".format(i))
            worker_cores = min(self.cpus_per_worker, cores_left)
            cores_left -= self.cpus_per_worker
            
            worker = mp.Process(target=self.handleTasks, args=(i, self.task_queue, self.out_queue, self.pid, start_core, worker_cores))

            self.workers.append(worker)
            start_core += self.cpus_per_worker

        for w in self.workers:
            w.start()

        for w in self.workers:
            w.join()

        log.info("{} : Exiting consumer process".format(os.getpid()))

class SUT(object):
    def __init__(self, num_proc, cpus_per_proc,library_path="", model_path=None, initial_core=0, batch_size=1, dataset_path=None, workers_per_proc=1, warmup=False, precision="int8", quantized_model=None, total_sample_count=1000, pad_inputs=False):
        self.library_path = library_path
        self.model_path = model_path
        self.num_proc = num_proc
        self.cpus_per_proc = cpus_per_proc
        self.initial_core = initial_core
        self.procs = [None] * self.num_proc
        self.workers_per_proc = workers_per_proc
        self.warmup = warmup
        self.total_workers = self.num_proc * self.workers_per_proc

        self.precision = precision
        self.quantized_model = quantized_model

        self.batch_size = batch_size
        self.dataset_path = dataset_path

        self.total_sample_count = total_sample_count
        self.pad_inputs = pad_inputs

        self.lock = mp.Lock()
        self.init_counter = mp.Value("i", 0)
        self.input_queue = mp.JoinableQueue()
        self.output_queue = mp.Queue()
        
        self.cv = mp.Condition(lock=self.lock)

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

        # Create processes
        self.createProcesses()

        # Start processes
        log.info("Starting processes")
        for proc in self.procs:
            proc.start()
        
        # Wait for all consumers to be ready (including if they're warming up)
        with self.cv:
            self.cv.wait_for(lambda : self.init_counter.value==self.num_proc)

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
            self.procs[proc_idx] = Consumer(self.library_path, self.model_path, self.precision, self.quantized_model, self.dataset_path, self.input_queue, self.output_queue, self.lock, self.cv, self.init_counter, proc_idx, start_core, self.cpus_per_proc, self.workers_per_proc, warmup=self.warmup, total_sample_count = self.total_sample_count, pad_inputs=self.pad_inputs)

            start_core += self.cpus_per_proc

    def issueQueries(self, query_samples):
        """ Receives queries and adds them to queue for processing"""
        # TODO: Implement Server logic in separate issueQuery

        num_samples = len(query_samples)
        ids = []        # Response Ids
        indexes = []    # Sample Indexes
        input_token_ids = []
        log.info('query length: {}'.format(num_samples))
        for i in range( num_samples):
            if len(ids)==self.batch_size:
                item = InputItem(ids, indexes)
                self.input_queue.put(item)
                ids = []
                indexes = []
                input_token_ids = []

            ids.append(query_samples[i].id)
            index = query_samples[i].index
            indexes.append(index)

        if ids:
            item = InputItem(ids, indexes)#, input_token_ids)
            self.input_queue.put(item)


