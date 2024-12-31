cmake_minimum_required(VERSION 3.13.4)

set(TARGET_NAME "monodomain")
project(${TARGET_NAME} CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/snapshot")
file(MAKE_DIRECTORY ${OUTPUT_DIR})

set(CNPY_DIR "$ENV{HOME}/cnpy/")
set(CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch")
set(CMAKE_INSTALL_RPATH "${CMAKE_PREFIX_PATH}/lib")


find_package(Torch REQUIRED)


find_package(deal.II 9.6.0 REQUIRED
        HINTS
        ${DEAL_II_DIR}
        ../
        ../../
        $ENV{DEAL_II_DIR}
        /opt/dealii/v9.6.0/lib/cmake/deal.II/
)


include_directories(
        ${CNPY_DIR}/include
        ${TORCH_INCLUDE_DIRS}
)

link_directories(
        ${CNPY_DIR}/lib
		$ENV{HOME}/zlib/lib/
)

set(TARGET_SRC
        monodomain.cpp
        applied_current.cpp
        ionic.cpp
        utils.cpp
	 	torch_inference.cpp
)

add_compile_options(-fvisibility=hidden)
add_link_options(-fvisibility=hidden)


add_executable(${PROJECT_NAME} ${TARGET_SRC})
target_link_libraries(${PROJECT_NAME}
        ${DEAL_II_LIBRARIES}
        ${TORCH_LIBRARIES}
        cnpy
        z
)

deal_ii_initialize_cached_variables()
deal_ii_setup_target(${PROJECT_NAME})
