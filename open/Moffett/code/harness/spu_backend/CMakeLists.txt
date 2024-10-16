
set(SOLA_RT_INC /usr/local/sola/include)
set(SOLA_RT_LIB /usr/local/sola/lib)

find_library(SOLA NAMES sola HINTS ${SOLA_RT_LIB})
find_library(ENGINE_LLM NAMES engine_llm HINTS ${SOLA_RT_LIB})

add_library(spu_backend
        src/server.cpp
        src/timer.cpp
        src/numa.cpp
        src/multithreading.cpp
        src/device.cpp
        src/device_gptj.cpp
        src/mf_sola.cpp
        src/model_executor.cpp
        )

target_link_libraries(spu_backend
        hwloc
        glog
        ${SOLA}
        ${ENGINE_LLM}
        OpenMP::OpenMP_CXX
        )

target_include_directories(spu_backend
        PUBLIC
        ${LOADGEN_INCLUDE_DIR}
        ${SOLA_RT_INC}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common>
        )
