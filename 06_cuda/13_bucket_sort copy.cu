#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void make_bucket(int *key, int *bucket, int num, int range){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < num) atomicAdd(&bucket[key[tid]],1);
}
__global__ void sort(int *key, int *bucket, int num, int range){
  int tid = threadIdx.x;
  int j = bucket[tid];
  for (int k=1; k<8; k<<=1) {
    int n = __shfl_up_sync(0xffffffff, j, k);
    if (tid >= k) j += n; 
  }
  j -= bucket[tid];
  for (; bucket[tid]>0; bucket[tid]--) key[j++] = tid;
}

int main() {
  int n = 50;
  int range = 5;
  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  // std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *bucket;
  cudaMallocManaged(&bucket, range*sizeof(int));
  make_bucket<<<(n + 128 - 1)/128 , 128>>>(key, bucket, n,range);
  sort<<<1, range>>>(key, bucket, n,range);
  cudaDeviceSynchronize();
  // std::vector<int> bucket(range); 
  // for (int i=0; i<range; i++) {
  // }
  // for (int i=0; i<n; i++) {
  //   bucket[key[i]]++;
  // }
  // for (int i=0, j=0; i<range; i++) {
  //   for (; bucket[i]>0; bucket[i]--) {
  //     key[j++] = i;
  //   }
  // }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}