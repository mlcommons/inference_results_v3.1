
#include "loadgen_plugin.h"
#include <test_settings.h>
#include <cstring>
#include <cstdio>
#include <iostream>

#ifdef MLPERF_VERBOSE_BUILD
#define MLPERF_VERBOSE(fmt) (printf(fmt));
#else
#define MLPERF_VERBOSE(fmt, ...) 
#endif

template <int C, int D>
struct ImgNet {
    size_t num_images;
    size_t image_size;
    int8_t (*data)[C][D][D]; // [num_images][num_channels][image_dim][image_dim]
};

typedef ImgNet<3, 224> ImageNet;

ImageNet* load_8bit_imagenet(const char* filename, const size_t num_images);

std::string SUT_NAME = "DaVinci_M.2";
std::string RESNET = "resnet";


mlperf::TestSettings MLPERF_TEST_SETTINGS;
mlperf::LogSettings MLPERF_LOG_SETTINGS;

SiMaSUT MLPERF_SUT;
SiMaQSL MLPERF_QSL;

ImageNet* MLPERF_IMAGENET; 

pthread_cond_t next_example_cv;
pthread_mutex_t next_example_mux;
size_t next_example_ready;
size_t next_example_size;
int8_t* next_example_ptr;
std::vector<mlperf::ResponseId> next_example_ids;

pthread_t loadgen_thread;

uint32_t SINGLE_STREAM_MODE = 0;
uint32_t MULTI_STREAM_MODE = 1;
uint32_t OFFLINE_MODE = 2;
// 0 = single stream
// 1 = multi stream
// 2 = offline
uint32_t mlperf_mode;

static int64_t sample_index = 0;

uint32_t RUN_TYPE_PERFORMANCE = 0;
uint32_t RUN_TYPE_ACCURACY = 1;
uint32_t run_type;

bool mlperf_toy_mode;
#define OFFLINE_BATCH_SIZE 24
int64_t offline_index;
int64_t offline_queries_completed;

int8_t *multistream_buffer;

std::vector<mlperf::QuerySampleResponse> responses;

size_t last_seen = 1;


void setup_config(uint32_t mode, uint32_t run_type, bool toy_mode) {
    std::string mode_str;
    if (mode == SINGLE_STREAM_MODE) {
        mode_str = "SingleStream";
    } else if (mode == MULTI_STREAM_MODE) {
        mode_str = "MultiStream";
    } else if (mode == OFFLINE_MODE) {
        mode_str = "Offline";
    }
    MLPERF_TEST_SETTINGS.FromConfig("mlperf.conf", "resnet50", mode_str);
    printf("Loaded mlperf.conf\n");
    MLPERF_TEST_SETTINGS.FromConfig("user.conf", "resnet50", mode_str);
    printf("Loaded user.conf\n");
    if (mode == SINGLE_STREAM_MODE) {
        MLPERF_TEST_SETTINGS.scenario = TestScenario::SingleStream;
        printf("Running in SingleStream Mode\n");
    } else if (mode == MULTI_STREAM_MODE) {
        MLPERF_TEST_SETTINGS.scenario = TestScenario::MultiStream;
        printf("Running in MultiStream Mode\n");
    } else if (mode == OFFLINE_MODE) {
        MLPERF_TEST_SETTINGS.scenario = TestScenario::Offline;
        printf("Running in Offline Mode\n");
    }
}

/* Initialize MLPerf ImageNet from an ImageNet filename.
 */
int mlperf_initialize(char* imagenet_filename, uint32_t mode, uint32_t run_type, bool toy_mode) {
    mlperf_mode = mode;
    std::cout << "SUT Name: " << MLPERF_SUT.Name() << std::endl;
    std::cout << "SQL Name: " << MLPERF_QSL.Name() << std::endl;

    multistream_buffer = new int8_t[3 * 224 * 224 * 8];

    if (toy_mode) {
        printf("** RUNNING IN TOY MODE **\n");
        MLPERF_QSL.num_images = 1000;
    } else {
        MLPERF_QSL.num_images = 50000;
    }
    MLPERF_TEST_SETTINGS.min_query_count = MLPERF_QSL.num_images;
    MLPERF_TEST_SETTINGS.min_duration_ms = 30000;

    setup_config(mode, run_type, toy_mode);
    // if (mode == SINGLE_STREAM_MODE) {
    //     MLPERF_TEST_SETTINGS.scenario = TestScenario::SingleStream;
    // } else if (mode == MULTI_STREAM_MODE) {
    //     MLPERF_TEST_SETTINGS.scenario = TestScenario::MultiStream;
    // } else if (mode == OFFLINE_MODE) {
    //     MLPERF_TEST_SETTINGS.scenario = TestScenario::Offline;
    // }
    if (run_type == RUN_TYPE_ACCURACY) {
        MLPERF_TEST_SETTINGS.mode = TestMode::AccuracyOnly;
        printf("Running Accuracy Mode.\n");
    } else if (run_type == RUN_TYPE_PERFORMANCE) {
        MLPERF_TEST_SETTINGS.mode = TestMode::PerformanceOnly;
        printf("Running Performance Mode.\n");
    }
    MLPERF_IMAGENET = load_8bit_imagenet(imagenet_filename, MLPERF_QSL.num_images);
    offline_index = -1;
    printf("Running Mode: %d\n", mlperf_mode);

    next_example_ready = last_seen;

    ImageNet* imagenet = MLPERF_IMAGENET;
    printf("Checking Values...\n");
    printf("imagenet[0][0][0][0] = %d\n", imagenet->data[0][0][0][0]);
    printf("imagenet[1][0][0][0] = %d\n", imagenet->data[1][0][0][0]);
    printf("imagenet[0][1][0][0] = %d\n", imagenet->data[0][1][0][0]);
    printf("imagenet[0][0][1][0] = %d\n", imagenet->data[0][0][1][0]);
    printf("imagenet[0][0][0][1] = %d\n", imagenet->data[0][0][0][1]);
    printf("imagenet[2][2][2][2] = %d\n", imagenet->data[2][2][2][2]);
    printf("imagenet[50][2][2][2] = %d\n", imagenet->data[50][2][2][2]);

    pthread_cond_init(&next_example_cv, NULL);
    pthread_mutex_init(&next_example_mux, NULL);
    printf("** RUNNING FOR MINIMUM %.1f MINUTES\n", MLPERF_TEST_SETTINGS.min_duration_ms / 1000.0 / 60.0);
    return 1;
}

uint32_t get_batch_size() {
    if (mlperf_mode == SINGLE_STREAM_MODE) {
        return 1;
    } else if (mlperf_mode == MULTI_STREAM_MODE){
        return 8;
    } else if (mlperf_mode == OFFLINE_MODE) {
        return OFFLINE_BATCH_SIZE;
    }
}

uint64_t get_batch_buffer_size() {
    return get_batch_buffer_size(mlperf_mode);
}

uint64_t get_batch_buffer_size(uint32_t mode) {
    uint64_t img_size = 3 * 224 * 224;
    if (mode == SINGLE_STREAM_MODE) {
        return 1 * img_size;
    } else if (mode == MULTI_STREAM_MODE) {
        return 8 * img_size;
    } else if (mode == OFFLINE_MODE) {
        return OFFLINE_BATCH_SIZE * img_size;
    } else {
        printf("ERROR: UNKNOWN MODE %ld\n", mode);
    }
}


void* _internal_start(void* _) {
    printf("--- MLPerf Clock Start\n");
    mlperf::StartTest(&MLPERF_SUT, &MLPERF_QSL, MLPERF_TEST_SETTINGS, MLPERF_LOG_SETTINGS);
    printf("--- MLPerf Clock Stop\n");
    printf(":::: StartTest(...) Returned.\n");
    pthread_mutex_lock(&next_example_mux);
    printf(":::: StartTest(...) Returned - signaled\n");
    next_example_ready = 0;
    pthread_cond_signal(&next_example_cv);
    pthread_mutex_unlock(&next_example_mux);
    return NULL;
}

int mlperf_start() {
    offline_queries_completed = 0;
    pthread_create(&loadgen_thread, NULL, _internal_start, NULL);
    return 1;
}

/* Blocks for the next MLPerf example. Returns NULL if no next example.
 */
int8_t* mlperf_block_for_next_example(size_t* bytes) {
    if (mlperf_mode == SINGLE_STREAM_MODE) {
        // printf("About to block for next query...\n");
        pthread_mutex_lock(&next_example_mux);
        // printf("Waiting for one after #%ld (next=%ld)\n", last_seen, next_example_ready);
        while (next_example_ready == last_seen && next_example_ready != 0) {
            pthread_cond_wait(&next_example_cv, &next_example_mux);
        }
        last_seen = next_example_ready;
        // printf("Got Next One: #%ld\n", last_seen);

        if (next_example_ready == 0) {
            printf("mlperf_block_for_next_example: REACHED END OF STREAM.\n");
            *bytes = 0;
            return NULL;
        }
        pthread_mutex_unlock(&next_example_mux);
        // printf("Unblocked after next query!\n");
        // We know we are safe to access size_t next_example_size and int8_t* next_example_ptr

        // TODO?
        *bytes = MLPERF_IMAGENET->image_size;
        return next_example_ptr;
    } else if (mlperf_mode == MULTI_STREAM_MODE) {
                // printf("About to block for next query...\n");
        pthread_mutex_lock(&next_example_mux);
        // printf("Waiting for one after #%ld (next=%ld)\n", last_seen, next_example_ready);
        while (next_example_ready == last_seen && next_example_ready != 0) {
            pthread_cond_wait(&next_example_cv, &next_example_mux);
        }
        last_seen = next_example_ready;
        if (next_example_ready == 0) {
            printf("mlperf_block_for_next_example: REACHED END OF STREAM.\n");
            *bytes = 0;
            return NULL;
        }
        pthread_mutex_unlock(&next_example_mux);
        // printf("Unblocked after next query!\n");
        // We know we are safe to access size_t next_example_size and int8_t* next_example_ptr
        *bytes = MLPERF_IMAGENET->image_size * 8;
        return next_example_ptr;
    } else if (mlperf_mode == OFFLINE_MODE) {
        if (offline_index < 0) {
            MLPERF_VERBOSE("Getting First Block of Offline Examples.\n");
            pthread_mutex_lock(&next_example_mux);
            while (next_example_ready == last_seen && next_example_ready != 0) {
                pthread_cond_wait(&next_example_cv, &next_example_mux);
            }
            pthread_mutex_unlock(&next_example_mux);
            // Go to the start of the array
            offline_index = 0;
        }
        if (offline_index >= MLPERF_QSL.num_images) {
            printf("mlperf_block_for_next_example: REACHED END OF STREAM.\n");
            return NULL;
            *bytes = 0;
        }
        int8_t *buf_start = &MLPERF_IMAGENET->data[offline_index][0][0][0];
        // This will apparently read off the end of the buffer... but the buffer was allocated with extra room.
        // This pads out the last batch to the full size expected.
        *bytes = MLPERF_IMAGENET->image_size * OFFLINE_BATCH_SIZE;
        offline_index += OFFLINE_BATCH_SIZE;
        MLPERF_VERBOSE("Returning Offline Examples.\n");
        return buf_start;
    }
}

/* Blocks for the next MLPerf example. Returns NULL if no next example.
 */
void mlperf_example_complete(std::vector<int32_t> &arg_max) {
    // printf("QueryComplete(...) Thread=%ld\n", std::this_thread::get_id());
    // printf("Location of next_example_ids: %lx\n", &next_example_ids);
    // printf("Length of next_example_ids: %lld\n", next_example_ids.size());

    MLPERF_VERBOSE("Marking Query Complete.\n");

    if (mlperf_mode == SINGLE_STREAM_MODE) {
        for (size_t i = 0; i < next_example_ids.size(); i++) {
            QuerySampleResponse resp;
            resp.id = next_example_ids[i];
            resp.data = reinterpret_cast<uintptr_t>(&arg_max[0]);
            resp.size = sizeof(int32_t);
            // printf("Issuing query complete %ld.\n", resp.id);
            QuerySamplesComplete(&resp, 1);
            // printf("QuerySamplesComplete Finished.\n");
        }
    } else if (mlperf_mode == MULTI_STREAM_MODE) {
        QuerySampleResponse resp[8];
        for (size_t i = 0; i < next_example_ids.size(); i++) {
            resp[i].id = next_example_ids[i];
            resp[i].data = reinterpret_cast<uintptr_t>(&arg_max[i]);
            resp[i].size = sizeof(int32_t);
        }
        QuerySamplesComplete(resp, next_example_ids.size());
    } else if (mlperf_mode == OFFLINE_MODE) {
      QuerySampleResponse resp[24];
      size_t bs = 24;
      int32_t num_left = next_example_ids.size() - offline_queries_completed;
      if (num_left < 24) {
        bs = num_left;
      }
      for (size_t i = 0; i < bs; i++) {
            resp[i].id = next_example_ids[offline_queries_completed + i];
            resp[i].data = reinterpret_cast<uintptr_t>(&arg_max[i]);
            resp[i].size = sizeof(int32_t);
        }
        QuerySamplesComplete(resp, bs);
        offline_queries_completed += bs;
    }
    
    MLPERF_VERBOSE("Done Marking Query Complete.\n");
}

int64_t mlperf_get_sample_index() {
    return sample_index;
}

// Called when loadgen wants to give us a new example to process.
void SiMaSUT::IssueQuery(const std::vector<QuerySample>& samples) {
    // Loadgen should only call this after we completed the last one. 
    // So we can assume we have access to the next_example_* fields.
    // printf("IssueQuery(...) Thread=UNknown\n");
    // printf("Location of next_example_ids: %lx\n", &next_example_ids);

    pthread_mutex_lock(&next_example_mux);
    next_example_ids.clear();
    for (size_t i = 0; i < samples.size(); i++) {
        // printf("Issuing query on: %ld\n", samples[i].id);
        next_example_ids.push_back(samples[i].id);
    }
    // printf("issued next_example_ids: %lld.\n", next_example_ids.size());
    if (mlperf_mode == SINGLE_STREAM_MODE) {
        // printf("Handling Single STream mode\n");
        next_example_ptr = &MLPERF_IMAGENET->data[samples[0].index][0][0][0];
        next_example_ready = last_seen + 1;
        sample_index = samples[0].index;
	// printf("\nNext sample index : %d\n", samples[0].index);
        // printf("Signaling Next: #%ld\n", next_example_ready);
    } else if (mlperf_mode == MULTI_STREAM_MODE) {
        // consolidate all 8 images into a single buffer
        for (size_t i = 0; i < next_example_ids.size(); i++) {
            memcpy(multistream_buffer + i * MLPERF_IMAGENET->image_size, &MLPERF_IMAGENET->data[samples[i].index][0][0][0], MLPERF_IMAGENET->image_size);
        }
        next_example_ready = last_seen + 1;
        next_example_ptr = multistream_buffer;
    } else if (mlperf_mode == OFFLINE_MODE) {
        // printf("Handling Offline Mode\n");
        // we should assume we have a long list of samples and we work through them sequentially.
        // Just return everything and let another function batch it.
        MLPERF_VERBOSE("IssueQuery(...): Only Offline Call Made.\n")
        printf("Issue Offline Query w/ %lld samples.\n", samples.size());
        next_example_ptr = &MLPERF_IMAGENET->data[0][0][0][0];
        next_example_ready = last_seen + 1;
        offline_index = -1;
        offline_queries_completed = 0;
        responses.clear();
    }
    pthread_cond_signal(&next_example_cv);
    pthread_mutex_unlock(&next_example_mux);
}

const std::string& SiMaSUT::Name() {
    return SUT_NAME;
}

void SiMaSUT::FlushQueries() {
}

const std::string& SiMaQSL::Name() {
return RESNET;
}

template <int C, int D>
ImgNet<C, D>* load_8bit_images(const char* filename, const size_t num_images) {
    printf("Loading file: %s\n", filename);
    FILE *fh = fopen(filename, "r");
    const size_t total_elements = num_images * C * D * D;
    int8_t (*data)[C][D][D] = new int8_t[num_images][C][D][D];
    printf("Loading %ld bytes...\n", total_elements);
    size_t bytes_read = fread(data, sizeof(int8_t), total_elements, fh);
    fclose(fh);
    ImgNet<C, D> *imgnet;
    imgnet = new ImgNet<C, D>();
    imgnet->num_images = num_images;
    imgnet->data = data;
    imgnet->image_size = C * D * D;
    printf("Done Loading.\n");
    return imgnet;
}

ImageNet* load_8bit_imagenet(const char* filename, const size_t num_images) {
    return load_8bit_images<3, 224>(filename, num_images);
}

