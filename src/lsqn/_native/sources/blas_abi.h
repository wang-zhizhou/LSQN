#ifndef LSQN_BLAS_ABI_H
#define LSQN_BLAS_ABI_H
#include <stdint.h>
#if LSQN_BLAS_BITS == 64
typedef int64_t lsqn_blas_int;
#elif LSQN_BLAS_BITS == 32
typedef int32_t lsqn_blas_int;
#else
#error "LSQN_BLAS_BITS must be 32 or 64"
#endif
#endif
