cmake_minimum_required(VERSION 3.13.4)
set(TARGET "monodomain")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/snapshot")
set(CNPY_DIR "/u/f/folivo/cnpy/")
file(MAKE_DIRECTORY ${OUTPUT_DIR})

project(${TARGET} CXX)

find_package(deal.II 9.6.0 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR} /opt/dealii/v9.6.0/lib/cmake/deal.II/
)

include_directories(${CNPY_DIR}/include)
link_directories(${CNPY_DIR}/lib)
link_directories(/u/f/folivo/zlib-1.3.1/lib/)
set(TARGET_SRC ${TARGET}.cpp applied_current.cpp ionic.cpp utils.cpp)

deal_ii_initialize_cached_variables()
deal_ii_invoke_autopilot()
# Link libraries
target_link_libraries(${PROJECT_NAME} cnpy z)

