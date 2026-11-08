/* Original arithmetic retained; BLAS integer ABI selected at compilation. */
#include "blas_abi.h"
#include <stdint.h>
#include <stddef.h>
#include <math.h>

typedef double (*dot64_t)(const lsqn_blas_int, const double *, const lsqn_blas_int,
                         const double *, const lsqn_blas_int);
typedef void (*gemv64_t)(int, int, lsqn_blas_int, lsqn_blas_int, double, const double *,
                        lsqn_blas_int, const double *, lsqn_blas_int, double, double *, lsqn_blas_int);

/* Column-major buffers, the same two reorthogonalization passes as SCB. */
void scb_project(const double *u, double *x, double *work,
                 int64_t n, int64_t m, gemv64_t gemv) {
    if (!m) return;
    for (int pass=0;pass<2;++pass) {
        gemv(102,112,n,m,1.,u,n,x,1,0.,work,1);
        gemv(102,111,n,m,-1.,u,n,work,1,1.,x,1);
    }
}

void scb_protect(const double *q, const double *s, double *z, double *w,
                 double *work, int64_t n, int64_t m, gemv64_t gemv) {
    if (!m) return;
    for (int pass=0;pass<2;++pass) {
        gemv(102,112,n,m,1.,q,n,z,1,0.,work,1);
        gemv(102,111,n,m,-1.,q,n,work,1,1.,z,1);
        gemv(102,111,n,m,-1.,s,n,work,1,1.,w,1);
    }
}

void scb_lowrank(const double *a, const double *c, const double *x,
                 double *out, double *work, int64_t n, int64_t m,
                 double beta, gemv64_t gemv) {
    for (int64_t i=0;i<n;++i) out[i]=beta*x[i];
    if (!m) return;
    gemv(102,112,n,m,1.,c,n,x,1,0.,work,1);
    gemv(102,111,n,m,1.,a,n,work,1,1.,out,1);
}

void scb_dense_policy(const double *const *actions, const int64_t *policy,
                      const double *x, double *out, size_t n, dot64_t dot) {
    for (size_t i=0;i<n;++i)
        out[i]=dot((int64_t)n, actions[policy[i]]+i*n, 1, x, 1);
}

void scb_dense_rows(const double *p, const int64_t *rows, const double *x,
                   double *out, size_t n, size_t m, dot64_t dot) {
    for (size_t i=0;i<m;++i)
        out[i]=dot((int64_t)n,p+rows[i]*n,1,x,1);
}

void scb_sparse_policy(const double *const *data, const int32_t *const *indices,
                       const int32_t *const *indptr, const int64_t *policy,
                       const double *x, double *out, size_t n) {
    for (size_t i=0;i<n;++i) {
        int64_t a=policy[i];
        double value=0.;
        for (int32_t k=indptr[a][i];k<indptr[a][i+1];++k)
            value+=data[a][k]*x[indices[a][k]];
        out[i]=value;
    }
}

void scb_sparse_j(const double *const *data, const int32_t *const *indices,
                   const int32_t *const *indptr, const int64_t *policy,
                   const double *x, double *out, size_t n, double gamma) {
    for (size_t i=0;i<n;++i) {
        int64_t a=policy[i];
        double value=0.;
        for (int32_t k=indptr[a][i];k<indptr[a][i+1];++k)
            value+=data[a][k]*x[indices[a][k]];
        out[i]=x[i]-gamma*value;
    }
}

void scb_sparse_certificate(const double *const *data, const int32_t *const *indices,
                            const int32_t *const *indptr, const int64_t *policy,
                            const uint8_t *mask, const double *f, const double *x,
                            const double *jx, double *out, size_t n, size_t na,
                            double gamma) {
    for (size_t i=0;i<n;++i) out[i]=f[i]*jx[i];
    for (size_t a=0;a<na;++a) {
        for (size_t i=0;i<n;++i) {
            if (policy[i]==(int64_t)a || !mask[a*n+i]) continue;
            double value=0.;
            for (int32_t k=indptr[a][i];k<indptr[a][i+1];++k)
                value+=data[a][k]*x[indices[a][k]];
            value=f[i]*(x[i]-gamma*value);
            if (value>out[i] || isnan(value)) out[i]=value;
        }
    }
}
