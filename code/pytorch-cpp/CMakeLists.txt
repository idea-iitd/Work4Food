cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(TARGET main)

project(${TARGET})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3 -fopenmp")

set(INC ../include)
set(INC_HHL ../include/msr_hhl)

set(DEPS_UTIL ${INC}/food_data_util.cpp ${INC}/graph_file_util.cpp ${INC}/graph_util.cpp ${INC}/global.cpp ${INC}/simulation_util.cpp ${INC}/constants.cpp ${INC}/hhl_query.cpp )
set(DEPS_CLASS ${INC}/config.cpp ${INC}/event.cpp ${INC}/order.cpp ${INC}/vehicle.cpp ${INC}/dsu.cpp ${INC}/gaussian_process.cpp)
set(DEPS_ALGO ${INC}/routeplan.cpp ${INC}/route_recommendation.cpp ${INC}/hungarian.cpp ${INC}/vehicle_assignment.cpp ${INC}/vehicle_assignment_baseline.cpp ${INC}/mdrp_baseline.cpp )

include_directories(
    ${INC}
    ${INC_HHL}
    )

add_executable(${TARGET} ${INC}/../main.cpp ${DEPS_UTIL} ${DEPS_CLASS} ${DEPS_ALGO})
# add_executable(${TARGET} main.cpp ${DEPS_UTIL} ${DEPS_CLASS} ${DEPS_ALGO})


target_link_libraries(${TARGET} "${TORCH_LIBRARIES}")
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 14)