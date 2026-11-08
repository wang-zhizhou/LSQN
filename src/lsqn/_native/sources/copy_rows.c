/* Original arithmetic retained; BLAS integer ABI selected at compilation. */
#include "blas_abi.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* Exact row copies; no floating-point arithmetic or model approximation. */
void scb_copy_policy_rows(double *output, const double * const *actions,
                          const int64_t *policy, const uint8_t *changed,
                          size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (changed[i]) {
            memcpy(output + i*n, actions[policy[i]] + i*n, n*sizeof(double));
        }
    }
}
