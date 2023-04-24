#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
 __m256 mvec  = _mm256_load_ps(m);
  for(int i=0; i<N; i++) {
   // for(int j=0; j<N; j++) {
    __m256 xi = _mm256_set1_ps(x[i]);
    __m256 yi = _mm256_set1_ps(y[i]);
    __m256 xj = _mm256_load_ps(x);
    __m256 yj = _mm256_load_ps(y);
    __m256 rx = _mm256_sub_ps(xi,xj);
    __m256 ry = _mm256_sub_ps(yi,yj);
    __m256 rr = _mm256_add_ps(_mm256_mul_ps(rx,rx),_mm256_mul_ps(ry,ry));

    __m256 mask = _mm256_cmp_ps(rr, _mm256_setzero_ps(), _CMP_GT_OQ); 
    __m256 ri = _mm256_rsqrt_ps(rr);  
    __m256 fxi = _mm256_mul_ps(mvec, rx);
    __m256 fxt = _mm256_mul_ps(mvec, rx);
    fxt = _mm256_mul_ps(fxt,ri);
    fxt = _mm256_mul_ps(fxt,ri);
    fxt = _mm256_mul_ps(fxt,ri);
    fxt = _mm256_blendv_ps(_mm256_setzero_ps(),fxt,mask);

    __m256 fyt = _mm256_mul_ps(mvec,ry);
    fyt = _mm256_mul_ps(fyt,ri);
    fyt = _mm256_mul_ps(fyt,ri);
    fyt = _mm256_mul_ps(fyt,ri);
    fyt = _mm256_blendv_ps(_mm256_setzero_ps(),fyt,mask);

    __m256 fxvec = _mm256_permute2f128_ps(fxt,fxt,1);
    fxvec = _mm256_add_ps(fxvec,fxt);
    fxvec = _mm256_hadd_ps(fxvec,fxvec);
    fxvec = _mm256_hadd_ps(fxvec,fxvec);
    fxvec = _mm256_mul_ps(fxvec,_mm256_set1_ps(-1));
    _mm256_store_ps(fx, fxvec);

    __m256 fyvec = _mm256_permute2f128_ps(fyt,fyt,1);
    fyvec = _mm256_add_ps(fyvec,fxt);
    fyvec = _mm256_hadd_ps(fyvec,fyvec);
    fyvec = _mm256_hadd_ps(fyvec,fyvec);
    fyvec = _mm256_mul_ps(fyvec,_mm256_set1_ps(-1));
    _mm256_store_ps(fy, fyvec);

    printf("%d %g %g\n",i,fx[i],fy[i]);
      // if(i != j) {
      //   float rx = x[i] - x[j];
      //   float ry = y[i] - y[j];
      //   float r = std::sqrt(rx * rx + ry * ry);
      //   fx[i] -= rx * m[j] / (r * r * r);
      //   fy[i] -= ry * m[j] / (r * r * r);
      // }
    // }
    // printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}