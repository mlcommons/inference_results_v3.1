cmake_minimum_required(VERSION 3.10)
project(harness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
# asan
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
#set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fno-omit-frame-pointer -fsanitize=address")

add_definitions(-DLOADGEN_INCLUDE_DIR=${LOADGEN_INCLUDE_DIR})
IF(${SERVER_MODE})
    add_definitions(-DSERVER_MODE=${SERVER_MODE})
ENDIF()
message("Server mode: ${SERVER_MODE}")
    # Find the static Loadgen library
unset(LOADGEN_LIB CACHE)
find_library(LOADGEN_LIB NAMES libmlperf_loadgen.a PATHS ${LOADGEN_LIB_DIR})
find_package(OpenMP)
# find_package(pybind11 REQUIRED)

# Set the path to the LWIS library
unset(LWIS_INCLUDE_DIR CACHE)
set(LWIS_INCLUDE_DIR spu_backend/include)

message(${LOADGEN_LIB_DIR})
message(${LOADGEN_LIB})

add_subdirectory(spu_backend)

# Build the default harness which covers single_stream and offline scenarios on image benchmarks.
execute_process(COMMAND echo "Building default harness...")
add_executable(harness_default
    harness_default/main_default.cpp
)
set_target_properties(harness_default PROPERTIES LINK_FLAGS "-Wl,-rpath,/usr/local/lib,-rpath,/usr/local/lib64")
target_link_libraries(harness_default
    gflags
    glog
    spu_backend
    ${LOADGEN_LIB}
    pthread
    OpenMP::OpenMP_CXX
)

target_include_directories(harness_default
    PUBLIC
    ${LWIS_INCLUDE_DIR}
    ${LOADGEN_INCLUDE_DIR}
    common
)

#execute_process(COMMAND echo "Building bert harness...")
#add_executable(harness_bert
#    harness_bert/main_bert.cpp
#)
#link_libraries(${PYTHON_LIBRARY})
#target_link_libraries(harness_bert
#    -Wl,-rpath,${PYTHON_LIBRARY},-rpath,/usr/local/lib,-rpath,/usr/local/lib64
#    gflags
#    glog
#    spu_backend
#    ${LOADGEN_LIB}
#    pthread
#    python3.9
#)

#
#execute_process(COMMAND echo "Building retinanet harness...")
#add_executable(harness_retinanet
#    harness_retinanet/retinanet_postprocess.h
#    harness_retinanet/retinanet_postprocess.cpp
#    harness_retinanet/main_retinanet.cpp
#)
#link_libraries(${PYTHON_LIBRARY})
#target_link_libraries(harness_retinanet
#    gflags
#    glog
#    spu_backend
#    ${LOADGEN_LIB}
#    pthread
#    python3.9
#    pybind11::embed
#    OpenMP::OpenMP_CXX
#)
#
#target_include_directories(harness_retinanet
#    PUBLIC
#    ${LWIS_INCLUDE_DIR}
#    ${LOADGEN_INCLUDE_DIR}
#    common
#)

execute_process(COMMAND echo "Building gptj harness...")
add_executable(harness_gptj
    harness_gptj/main_gptj.cpp
)
set_target_properties(harness_gptj PROPERTIES LINK_FLAGS "-Wl,-rpath,/usr/local/lib,-rpath,/usr/local/lib64")
target_link_libraries(harness_gptj
    gflags
    glog
    spu_backend
    ${LOADGEN_LIB}
    pthread
    OpenMP::OpenMP_CXX
)

target_include_directories(harness_gptj
    PUBLIC
    ${LWIS_INCLUDE_DIR}
    ${LOADGEN_INCLUDE_DIR}
    common
)
