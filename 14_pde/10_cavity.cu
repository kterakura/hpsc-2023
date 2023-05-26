#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>

const int nx = 41;
const int ny = 41;
const int nt = 500;
const int nit = 50;
const double dx = 2.0 / (nx - 1);
const double dy = 2.0 / (ny - 1);
const double dt = 0.01;
const double rho = 1.0;
const double nu = 0.02;

__global__ void buildMesh(double* x, double* y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < nx) {
        x[i] = i * dx;
    }
    if (i < ny) {
        y[i] = i * dy;
    }
}

__global__ void initializeArrays(double* u, double* v, double* p, double* b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = j * nx + i;
    if (i < nx && j < ny) {
        u[idx] = 0.0;
        v[idx] = 0.0;
        p[idx] = 0.0;
        b[idx] = 0.0;
    }
}

__global__ void solve(double* u, double* v, double* p, double* b, double* x, double* y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = j * nx + i;
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        b[idx] = rho * (1.0 / dt * 
                        ((u[idx + 1] - u[idx - 1]) / (2 * dx) + (v[idx + nx] - v[idx - nx]) / (2 * dy)) -
                        ((u[idx + 1] - u[idx - 1]) / (2 * dx)) * ((u[idx + 1] - u[idx - 1]) / (2 * dx)) - 2.0 * ((u[idx + nx] - u[idx - nx]) / (2 * dy)) * 
                        ((v[idx + 1] - v[idx - 1]) / (2 * dx)) - ((v[idx + nx] - v[idx - nx]) / (2 * dy)) * ((v[idx + nx] - v[idx - nx]) / (2 * dy)));
    }

    __syncthreads();

    for (int it = 0; it < nit; ++it) {
        double* pn = p;
        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
            p[idx] =
                (dy * dy * (pn[idx + 1] + pn[idx - 1]) +
                 dx * dx * (pn[idx + nx] + pn[idx - nx]) -
                 b[idx] * dx * dx * dy * dy) /
                (2.0 * (dx * dx + dy * dy));
        }
        __syncthreads();
        if (j == 0) {
            p[i] = p[nx + i];
            p[(ny - 1) * nx + i] = 0.0;
        }
        if (i == 0) {
            p[j * nx] = p[j * nx + 1];
            p[j * nx + nx - 1] = p[j * nx + nx - 2];
        }
        __syncthreads();
    }

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        double* un = u;
        double* vn = v;
        u[idx] = un[idx] - un[idx] * dt / dx * (un[idx] - un[idx - 1]) -
                  vn[idx] * dt / dy * (un[idx] - un[idx - nx]) -
                  dt / (2.0 * rho * dx) * (p[idx + 1] - p[idx - 1]) +
                  nu * dt / (dx * dx) * (un[idx + 1] - 2.0 * un[idx] + un[idx - 1]) +
                  nu * dt / (dy * dy) * (un[idx + nx] - 2.0 * un[idx] + un[idx - nx]);

        v[idx] = vn[idx] - un[idx] * dt / dx * (vn[idx] - vn[idx - 1]) -
                  vn[idx] * dt / dy * (vn[idx] - vn[idx - nx]) -
                  dt / (2.0 * rho * dy) * (p[idx + nx] - p[idx - nx]) +
                  nu * dt / (dx * dx) * (vn[idx + 1] - 2.0 * vn[idx] + vn[idx - 1]) +
                  nu * dt / (dy * dy) * (vn[idx + nx] - 2.0 * vn[idx] + vn[idx - nx]);
    }

    __syncthreads();
    if (j == 0){
        u[i] = 0.0;
        u[(ny - 1) * nx + i] = 1.0;
        v[i] = 0.0;
        v[(ny - 1) * nx + i] = 0.0;
    }

    if (i == 0) {
        u[j * nx] = 0.0;
        u[j * nx + nx - 1] = 0.0;
        v[j * nx] = 0.0;
        v[j * nx + nx - 1] = 0.0;
    }
}

int main() {
    double* x, *y, *u, *v, *p, *b;
    cudaMallocManaged(&x, nx * sizeof(double));
    cudaMallocManaged(&y, ny * sizeof(double));
    cudaMallocManaged(&u, nx * ny * sizeof(double));
    cudaMallocManaged(&v, nx * ny * sizeof(double));
    cudaMallocManaged(&p, nx * ny * sizeof(double));
    cudaMallocManaged(&b, nx * ny * sizeof(double));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x, (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    buildMesh<<<numBlocks, threadsPerBlock>>>(x, y);
    initializeArrays<<<numBlocks, threadsPerBlock>>>(u, v, p, b);

    for (int n = 0; n < nt; ++n) {
        solve<<<numBlocks, threadsPerBlock>>>(u, v, p, b, x, y);
        cudaDeviceSynchronize();

        // Plotting (using file output instead of matplotlib)
        std::ofstream outFile("output/" + std::to_string(n) + ".txt");
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = j * nx + i;
                outFile << x[i] << " " << y[j] << " " << p[idx] << " " << u[idx] << " " << v[idx] << std::endl;
            }
        }
        outFile.close();
    }

    cudaFree(x);
    cudaFree(y);
    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);

    return 0;
}