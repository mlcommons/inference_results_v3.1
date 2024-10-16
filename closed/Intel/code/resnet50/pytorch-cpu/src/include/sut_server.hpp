


#ifndef SUT_SERVER_H_
#define SUT_SERVER_H_

#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <utility>
#include <unistd.h>
#include <atomic>

#include <torch/torch.h>

#include <loadgen.h>
#include <query_sample.h>
#include <query_sample_library.h>
#include <system_under_test.h>
#include <test_settings.h>

#include "dataset.hpp"
#include "backend.hpp"
#include "item.hpp"
// #include "kmp_launcher.hpp"
// #include "sut.hpp"

class SUTServer : public mlperf::SystemUnderTest {
public:
    SUTServer(std::string rn50_part1,
        std::string rn50_part3,
        std::string full_model,
        std::string data_path,
        size_t total_sample_count,
        int num_instances,
        int cpus_per_instance,
        std::string mlperf_user_conf,
        int batch_size
    ); 

    SUTServer(std::string rn50_part1,
        std::string rn50_part3,
        std::string full_model,
        std::string data_path,
        size_t total_sample_count,
        int num_instances,
        int cpus_per_instance,
        std::string mlperf_user_conf,
	    int warmup_count,
        int batch_size
	) :  SUTServer(rn50_part1,
        rn50_part3,
        full_model,
        data_path,
        total_sample_count,
        num_instances,
        cpus_per_instance,
        mlperf_user_conf,
        batch_size
    	) {

	this->num_warmup_ = warmup_count;
	}
    ~SUTServer();

    void startSUT();

    const std::string& Name() override;

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;

    void FlushQueries() override;

    void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies);

    void setCompletion();

    // Instance thread is responsible for performing inference
    void instanceThread();

    // Creates Instance (for models)
    // TODO: Make flexible to share weights across all instances
    std::unique_ptr<Backend> createInstance();

    // Returns qsl object shared with loadgen
    mlperf::QuerySampleLibrary* GetQsl(){
        return ds_;
    }

    // Perform warmup
    void doWarmup();

    // For batching server queries
    void QueryBatchingThread();

private:

    // Add necessary objects

    Dataset* ds_;
    std::unique_ptr<Backend> backend_;
    int batch_size_=4;
    at::Tensor input_tensor = torch::zeros({batch_size_, Start_In_C, Start_I_H, Start_I_W}).to(torch::kInt8).to(torch::MemoryFormat::ChannelsLast);
    at::Tensor backbone_output = torch::zeros({ batch_size_, 2048, 7, 7 }).to(torch::kInt8).to(torch::MemoryFormat::ChannelsLast);
    torch::jit::IValue outputs;
    std::vector<mlperf::QuerySampleResponse> responses;
    std::string rn50_part1_, rn50_part3_, full_model_;
    std::string data_path_;
    std::string name_ = "SUT_Server";

    size_t total_count_ = 1024;
    size_t perf_count_ = 1024;    

    int instances_ = 0;
    int cpus_per_instance_;

    
    std::vector<mlperf::QuerySampleIndex> batch_idxs_server;
    std::vector<mlperf::ResponseId> batch_resp_ids_server; 



    // Comm vars
    std::queue<Item> query_queue_;
    std::condition_variable cond_var_;
    std::mutex mutex_;
    bool completed_ = false;
    std::vector<std::thread> workers_;

    // Batcher thread objects. Takes query from loadgen, batches them and add to batched query queue
    std::deque<mlperf::QuerySample> minibatch_; // Rename??
    std::condition_variable batcher_cv_;
    std::mutex batcher_mutex_;
    int minibatch_size_ = 0;
    std::thread batcher_thread_;

    int start_core_ = 0;
    int instance_id_ = 0;
    mlperf::TestSettings settings_;
    int num_warmup_ = 0;

    //void doWarmup();
    bool start_warmup_ = false;
    std::atomic<int> idle_instances_;
    std::mutex warmup_mutex_;
    std::condition_variable w_cond_var_;


}; // SUTServer

#endif