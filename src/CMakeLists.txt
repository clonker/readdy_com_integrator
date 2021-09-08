project(bindings)

set(SRC main.cpp)

pybind11_add_module(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX ${READDY_DEPENDENT_LIBRARIES} readdy readdy_kernel_cpu)
target_include_directories(${PROJECT_NAME} PUBLIC "${READDY_INCLUDE_DIRS}" "${CPU_INCLUDE_DIR}")
