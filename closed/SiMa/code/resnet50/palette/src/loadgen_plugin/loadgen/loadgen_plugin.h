

#include <system_under_test.h>
#include <query_sample_library.h>
#include <loadgen.h>
#include <pthread.h>



#define MLPERF_TOY_BUILD TRUE

#ifdef MLPERF_TOY_BUILD
#define MLPERF_NUM_IMAGES 1000
#else
#define MLPERF_NUM_IMAGES 50000
#endif


using namespace mlperf;

// #define MLPERF_VERBOSE_BUILD TRUE




/* Initialize MLPerf ImageNet from an ImageNet filename.
 */
int mlperf_initialize(char* imagenet_filename, uint32_t mode, uint32_t run_type, bool toy_mode);

/* Initialize MLPerf ImageNet from an ImageNet filename.
 */
int mlperf_start();

uint32_t get_batch_size();
uint64_t get_batch_buffer_size();
uint64_t get_batch_buffer_size(uint32_t);

/* Blocks for the next MLPerf example. Returns NULL if no next example.
 */
int8_t* mlperf_block_for_next_example(size_t*);

/* Blocks for the next MLPerf example. Returns NULL if no next example.
 */
void mlperf_example_complete(std::vector<int32_t>&);

int64_t mlperf_get_sample_index();

class SiMaSUT : public SystemUnderTest {
public:

  const std::string& Name();

  void IssueQuery(const std::vector<QuerySample>& samples);

  virtual void FlushQueries();

};



class SiMaQSL : public QuerySampleLibrary {
 public:
  size_t num_images; 

  /// \brief A human readable name for the model.
  const std::string& Name();

  /// \brief Total number of samples in library.
  size_t TotalSampleCount() {
    return num_images;
  }

  /// \brief The number of samples that are guaranteed to fit in RAM.
  size_t PerformanceSampleCount() {
    return num_images;
  }

  /// \brief Loads the requested query samples into memory.
  /// \details Paired with calls to UnloadSamplesFromRam.
  /// In the MultiStream scenarios:
  ///   * Samples will appear more than once.
  ///   * SystemUnderTest::IssueQuery will only be called with a set of samples
  ///     that are neighbors in the vector of samples here, which helps
  ///     SUTs that need the queries to be contiguous.
  /// In all other scenarios:
  ///   * A previously loaded sample will not be loaded again.
  virtual void LoadSamplesToRam(
      const std::vector<QuerySampleIndex>& samples) {
      }

  /// \brief Unloads the requested query samples from memory.
  /// \details In the MultiStream scenarios:
  ///   * Samples may be unloaded the same number of times they were loaded;
  ///     however, if the implementation de-dups loaded samples rather than
  ///     loading samples into contiguous memory, it may unload a sample the
  ///     first time they see it unloaded without a refcounting scheme, ignoring
  ///     subsequent unloads. A refcounting scheme would also work, but is not
  ///     a requirement.
  /// In all other scenarios:
  ///   * A previously unloaded sample will not be unloaded again.
  void UnloadSamplesFromRam(
      const std::vector<QuerySampleIndex>& samples) {
      }
};
