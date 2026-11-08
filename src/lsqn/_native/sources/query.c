/* Original arithmetic retained; BLAS integer ABI selected at compilation. */
#include "blas_abi.h"
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <float.h>
#include <string.h>
typedef double (*dot_t)(lsqn_blas_int,const double*,lsqn_blas_int,const double*,lsqn_blas_int);
typedef void (*gemv_t)(int,int,lsqn_blas_int,lsqn_blas_int,double,const double*,lsqn_blas_int,const double*,lsqn_blas_int,double,double*,lsqn_blas_int);

/* Exactly two CGS passes; no basis direction, model term or test discarded. */
int scb_query(const uintptr_t *ptr,int64_t n,int64_t m,int64_t rank,int64_t pending,
              double beta,double gamma,double *out,gemv_t mv,dot_t dot) {
    const double *p=(void*)ptr[0],*jp=(void*)ptr[1],*U=(void*)ptr[2];
    const double *Q=(void*)ptr[3],*S=(void*)ptr[4],*H=(void*)ptr[5];
    const double *A=(void*)ptr[6],*C=(void*)ptr[7],*pa=(void*)ptr[8],*pc=(void*)ptr[9];
    const double *const *data=(void*)ptr[10];
    const int32_t *const *indices=(void*)ptr[11],*const *indptr=(void*)ptr[12];
    const int64_t *policy=(void*)ptr[13];
    double *u=out,*z=out+n,*w=out+2*n,*change=out+3*n,*work=out+4*n;
    memcpy(u,p,n*sizeof(double));
    if(m) for(int pass=0;pass<2;++pass) {
        mv(102,112,n,m,1.,U,n,u,1,0.,work,1);
        mv(102,111,n,m,-1.,U,n,work,1,1.,u,1);
    }
    double un=sqrt(dot(n,u,1,u,1)),pn=sqrt(dot(n,p,1,p,1));
    /* Rare coordinate fallback is delegated to the original implementation. */
    if(un<=DBL_EPSILON*fmax(1.,pn)) return 1;
    if(!isfinite(un)||un<=DBL_MIN) return 2;
    for(int64_t i=0;i<n;++i) u[i]/=un;
    if(!m) for(int64_t i=0;i<n;++i) z[i]=jp[i]/un;
    else for(int64_t i=0;i<n;++i) {
        int64_t a=policy[i];double v=0.;
        for(int32_t k=indptr[a][i];k<indptr[a][i+1];++k) v+=data[a][k]*u[indices[a][k]];
        z[i]=u[i]-gamma*v;
    }
    memcpy(w,u,n*sizeof(double));
    if(m) for(int pass=0;pass<2;++pass) {
        mv(102,112,n,m,1.,Q,n,z,1,0.,work,1);
        mv(102,111,n,m,-1.,Q,n,work,1,1.,z,1);
        mv(102,111,n,m,-1.,S,n,work,1,1.,w,1);
    }
    double zn=sqrt(dot(n,z,1,z,1));
    if(!isfinite(zn)||zn<=DBL_MIN)return 3;
    for(int64_t i=0;i<n;++i) {z[i]/=zn;w[i]/=zn;}
    if(H) {
        mv(102,111,n,n,1.,H,n,z,1,0.,change,1);
        if(pending) {
            mv(102,112,n,pending,1.,pc,n,z,1,0.,work,1);
            mv(102,111,n,pending,1.,pa,n,work,1,1.,change,1);
        }
    } else {
        for(int64_t i=0;i<n;++i)change[i]=beta*z[i];
        if(rank) {
            mv(102,112,n,rank,1.,C,n,z,1,0.,work,1);
            mv(102,111,n,rank,1.,A,n,work,1,1.,change,1);
        }
    }
    for(int64_t i=0;i<n;++i)change[i]=w[i]-change[i];
    return 0;
}
