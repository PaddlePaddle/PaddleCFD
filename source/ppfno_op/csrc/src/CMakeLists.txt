set(
        fused_segment_csr_csrc
        module.cpp
        select.cpp
        select.cu
)

pybind11_add_module(_C MODULE "${fused_segment_csr_csrc}")

target_link_libraries(_C PRIVATE ${PADDLE_LIBRARY} CUDA::cudart)
