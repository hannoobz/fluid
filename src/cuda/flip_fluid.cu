#include "flip_fluid.h"

#include <algorithm>
#include <cmath>

namespace flipgpu {

FlipFluid::FlipFluid(float density_, float width, float height, float spacing, 
                     float particle_radius, int max_particles,
                     int threads_1d, dim3 threads_2d) {
    
    threads1D = threads_1d;
    threads2D = threads_2d;

    density = density_;
    fNumX = int(std::floor(width / spacing)) + 1;
    fNumY = int(std::floor(height / spacing)) + 1;
    h = std::max(width / fNumX, height / fNumY);
    fInvSpacing = 1.0f / h;
    fNumCells = fNumX * fNumY;
    u.assign(fNumCells, 0.0f);
    v.assign(fNumCells, 0.0f);
    du.assign(fNumCells, 0.0f);
    dv.assign(fNumCells, 0.0f);
    prevU.assign(fNumCells, 0.0f);
    prevV.assign(fNumCells, 0.0f);
    p.assign(fNumCells, 0.0f);
    s.assign(fNumCells, 0.0f);
    cellType.assign(fNumCells, AIR_CELL);
    cellColor.assign(3 * fNumCells, 0.0f);
    maxParticles = max_particles;
    particlePosX.assign(maxParticles, 0.0f);
    particlePosY.assign(maxParticles, 0.0f);
    particleVelX.assign(maxParticles, 0.0f);
    particleVelY.assign(maxParticles, 0.0f);
    particleColorR.assign(maxParticles, 0.0f);
    particleColorG.assign(maxParticles, 0.0f);
    particleColorB.assign(maxParticles, 1.0f);
    particleDensity.assign(fNumCells, 0.0f);
    particleRestDensity = 0.0f;
    particleRadius = particle_radius;
    pInvSpacing = 1.0f / (2.2f * particleRadius);
    pNumX = int(std::floor(width * pInvSpacing)) + 1;
    pNumY = int(std::floor(height * pInvSpacing)) + 1;
    pNumCells = pNumX * pNumY;
    numCellParticles.assign(pNumCells, 0);
    firstCellParticle.assign(pNumCells + 1, 0);
    cellParticleIds.assign(maxParticles, 0);
    numParticles = 0;

    blocks1D_cells  = (fNumCells + threads1D - 1) / threads1D;
    blocks1D_pCells = (pNumCells + threads1D - 1) / threads1D;
    blocks2D_cells  = dim3((fNumX + threads2D.x - 1) / threads2D.x, 
                           (fNumY + threads2D.y - 1) / threads2D.y);

    cudaMalloc(&d_outSum,   sizeof(float));
	cudaMalloc(&d_outCount, sizeof(int));
    cudaMalloc(&d_u,               fNumCells      * sizeof(float));
    cudaMalloc(&d_v,               fNumCells      * sizeof(float));
    cudaMalloc(&d_du,              fNumCells      * sizeof(float));
    cudaMalloc(&d_dv,              fNumCells      * sizeof(float));
    cudaMalloc(&d_div,             fNumCells      * sizeof(float));
    cudaMalloc(&d_prevU,           fNumCells      * sizeof(float));
    cudaMalloc(&d_prevV,           fNumCells      * sizeof(float));
    cudaMalloc(&d_p,               fNumCells      * sizeof(float));
    cudaMalloc(&d_p_tmp,           fNumCells      * sizeof(float));
    cudaMalloc(&d_s,               fNumCells      * sizeof(float));
    cudaMalloc(&d_cellColor,       3 * fNumCells  * sizeof(float));
    cudaMalloc(&d_cellType,        fNumCells      * sizeof(int));
    cudaMalloc(&d_particlePosX,    maxParticles   * sizeof(float));
    cudaMalloc(&d_particlePosY,    maxParticles   * sizeof(float));
    cudaMalloc(&d_particleVelX,    maxParticles   * sizeof(float));
    cudaMalloc(&d_particleVelY,    maxParticles   * sizeof(float));
    cudaMalloc(&d_particleColorR,  maxParticles   * sizeof(float));
    cudaMalloc(&d_particleColorG,  maxParticles   * sizeof(float));
    cudaMalloc(&d_particleColorB,  maxParticles   * sizeof(float));
    cudaMalloc(&d_particleDensity, fNumCells      * sizeof(float));
    cudaMalloc(&d_numCellParticles,  pNumCells       * sizeof(int));
    cudaMalloc(&d_firstCellParticle,(pNumCells + 1)  * sizeof(int));
    cudaMalloc(&d_cellParticleIds,   maxParticles    * sizeof(int));
}

FlipFluid::~FlipFluid() {
    cudaFree(d_outSum);     cudaFree(d_outCount);
    cudaFree(d_u);          cudaFree(d_v);
    cudaFree(d_du);         cudaFree(d_dv);
    cudaFree(d_prevU);      cudaFree(d_prevV);
    cudaFree(d_p);          cudaFree(d_p_tmp);
    cudaFree(d_s);          cudaFree(d_div);
    cudaFree(d_cellColor);
    cudaFree(d_cellType);
    cudaFree(d_particlePosX);   cudaFree(d_particlePosY);
    cudaFree(d_particleVelX);   cudaFree(d_particleVelY);
    cudaFree(d_particleColorR); cudaFree(d_particleColorG); cudaFree(d_particleColorB);
    cudaFree(d_particleDensity);
    cudaFree(d_numCellParticles);
    cudaFree(d_firstCellParticle);
    cudaFree(d_cellParticleIds);
}

void FlipFluid::uploadToGPU() {
    cudaMemcpy(d_u,              u.data(),              fNumCells     * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,              v.data(),              fNumCells     * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_du,             du.data(),             fNumCells     * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dv,             dv.data(),             fNumCells     * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prevU,          prevU.data(),          fNumCells     * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prevV,          prevV.data(),          fNumCells     * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p,              p.data(),              fNumCells     * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s,              s.data(),              fNumCells     * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cellColor,      cellColor.data(),      3 * fNumCells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cellType,       cellType.data(),       fNumCells     * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_particlePosX,   particlePosX.data(),   maxParticles  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particlePosY,   particlePosY.data(),   maxParticles  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleVelX,   particleVelX.data(),   maxParticles  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleVelY,   particleVelY.data(),   maxParticles  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleColorR, particleColorR.data(), maxParticles  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleColorG, particleColorG.data(), maxParticles  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleColorB, particleColorB.data(), maxParticles  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleDensity,particleDensity.data(),fNumCells     * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numCellParticles,  numCellParticles.data(),  pNumCells      * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_firstCellParticle, firstCellParticle.data(), (pNumCells+1)  * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cellParticleIds,   cellParticleIds.data(),   maxParticles   * sizeof(int), cudaMemcpyHostToDevice);
}

__global__ void integrateParticles(
    float* posX, float* posY,
    float* velX, float* velY,
    float dt, float gravity, int numParticles){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    velY[i] += dt * gravity;
    posX[i] += velX[i] * dt;
    posY[i] += velY[i] * dt;
}

__global__ void countParticlesPerCell(const float* __restrict__ particlePosX,
    const float* __restrict__ particlePosY,
    int* __restrict__ numCellParticles,
    int numParticles, int pNumX, int pNumY, float pInvSpacing){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int xi = max(0, min(pNumX - 1, int(floorf(particlePosX[i] * pInvSpacing))));
    int yi = max(0, min(pNumY - 1, int(floorf(particlePosY[i] * pInvSpacing))));
    int cellNr = xi * pNumY + yi;

    atomicAdd(&numCellParticles[cellNr], 1);
}

__global__ void blockScan(const int* __restrict__ in,
                          int* __restrict__ out,
                          int* __restrict__ blockSums,
                          int n)
{
    extern __shared__ int s[];

    int tid   = threadIdx.x;
    int gid   = blockIdx.x * blockDim.x * 2 + tid;

    s[tid]              = (gid     < n) ? in[gid]              : 0;
    s[tid + blockDim.x] = (gid + blockDim.x < n) ? in[gid + blockDim.x] : 0;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x * 2; stride <<= 1) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < blockDim.x * 2)
            s[idx] += s[idx - stride];
        __syncthreads();
    }

    if (tid == 0) {
        if (blockSums) blockSums[blockIdx.x] = s[blockDim.x * 2 - 1];
        s[blockDim.x * 2 - 1] = 0;
    }
    __syncthreads();

    for (int stride = blockDim.x; stride >= 1; stride >>= 1) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < blockDim.x * 2) {
            int tmp        = s[idx - stride];
            s[idx - stride] = s[idx];
            s[idx]         += tmp;
        }
        __syncthreads();
    }

    if (gid     < n) out[gid]              = s[tid];
    if (gid + blockDim.x < n) out[gid + blockDim.x] = s[tid + blockDim.x];
}

__global__ void addBlockOffsets(int* data, const int* blockSums, int n)
{
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int offset = blockSums[blockIdx.x];

    if (gid     < n) data[gid]              += offset;
    if (gid + blockDim.x < n) data[gid + blockDim.x] += offset;
}

void prefixSum(const int* d_in, int* d_out, int n, int threads1D)
{
    if (n <= 0) return;

    int threadsPerBlock = threads1D; 
    int elemsPerBlock   = threadsPerBlock * 2;
    int numBlocks       = (n + elemsPerBlock - 1) / elemsPerBlock;
    size_t sharedBytes  = elemsPerBlock * sizeof(int);

    int* d_blockSums = nullptr;
    if (numBlocks > 1)
        cudaMalloc(&d_blockSums, numBlocks * sizeof(int));

    blockScan<<<numBlocks, threadsPerBlock, sharedBytes>>>(
        d_in, d_out, d_blockSums, n);

    if (numBlocks > 1) {
        int* d_scannedBlockSums;
        cudaMalloc(&d_scannedBlockSums, numBlocks * sizeof(int));
        prefixSum(d_blockSums, d_scannedBlockSums, numBlocks, threads1D);

        addBlockOffsets<<<numBlocks, threadsPerBlock>>>(
            d_out, d_scannedBlockSums, n);

        cudaFree(d_scannedBlockSums);
        cudaFree(d_blockSums);
    }
}

__global__ void fillCellParticles(
    float* particlePosX, float* particlePosY,
    int*   firstCellParticle, int* cellParticleIds,
    float  pInvSpacing, int pNumX, int pNumY,
    int    numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int xi     = max(0, min((int)floorf(particlePosX[i] * pInvSpacing), pNumX - 1));
    int yi     = max(0, min((int)floorf(particlePosY[i] * pInvSpacing), pNumY - 1));
    int cellNr = xi * pNumY + yi;

    int idx = atomicSub(&firstCellParticle[cellNr], 1) - 1;
    cellParticleIds[idx] = i;
}

__global__ void pushParticlesApart(
    float* particlePosX, float* particlePosY,
    float* particleColorR, float* particleColorG, float* particleColorB,
    int* firstCellParticle, int* cellParticleIds,
    float  particleRadius, float pInvSpacing,
    int    pNumX, int pNumY, int numParticles,
    float  colorDiffusionCoeff)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    const float minDist  = 2.0f * particleRadius;
    const float minDist2 = minDist * minDist;

    float px = particlePosX[i];
    float py = particlePosY[i];

    int pxi = (int)floorf(px * pInvSpacing);
    int pyi = (int)floorf(py * pInvSpacing);
    int x0  = max(pxi - 1, 0);
    int y0  = max(pyi - 1, 0);
    int x1  = min(pxi + 1, pNumX - 1);
    int y1  = min(pyi + 1, pNumY - 1);

    int checks = 0;

    for (int xi = x0; xi <= x1; ++xi) {
        for (int yi = y0; yi <= y1; ++yi) {
            int cellNr = xi * pNumY + yi;
            int firstI = firstCellParticle[cellNr];
            int lastI  = firstCellParticle[cellNr + 1];

            for (int j = firstI; j < lastI; ++j) {
                if (checks++ > 150) break;

                int idn = cellParticleIds[j];
                if (idn == i) continue;

                float qx = particlePosX[idn];
                float qy = particlePosY[idn];
                float dx = qx - px;
                float dy = qy - py;
                float d2 = dx * dx + dy * dy;
                if (d2 > minDist2 || d2 == 0.0f) continue;

                float d    = sqrtf(d2);
                float sFac = 0.5f * (minDist - d) / d;
                dx *= sFac;
                dy *= sFac;

                atomicAdd(&particlePosX[i], -dx);
                atomicAdd(&particlePosY[i], -dy);

                float c0r = particleColorR[i],   c1r = particleColorR[idn];
                float c0g = particleColorG[i],   c1g = particleColorG[idn];
                float c0b = particleColorB[i],   c1b = particleColorB[idn];
                float cr  = (c0r + c1r) * 0.5f;
                float cg  = (c0g + c1g) * 0.5f;
                float cb  = (c0b + c1b) * 0.5f;

                atomicAdd(&particleColorR[i], (cr - c0r) * colorDiffusionCoeff);
                atomicAdd(&particleColorG[i], (cg - c0g) * colorDiffusionCoeff);
                atomicAdd(&particleColorB[i], (cb - c0b) * colorDiffusionCoeff);

                px = particlePosX[i];
                py = particlePosY[i];
            }
        }
    }
}

__global__ void handleParticleCollisions(
    float* particlePosX, float* particlePosY,
    float* particleVelX, float* particleVelY,
    float obstacleX, float obstacleY, float obstacleRadius,
    float obstacleVelX, float obstacleVelY,
    float fInvSpacing, float particleRadius,
    int fNumX, int fNumY, int numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float hh       = 1.0f / fInvSpacing;
    float minDist2 = (obstacleRadius + particleRadius) * (obstacleRadius + particleRadius);
    float minX     = hh + particleRadius;
    float maxX     = (fNumX - 1) * hh - particleRadius;
    float minY     = hh + particleRadius;
    float maxY     = (fNumY - 1) * hh - particleRadius;

    float x  = particlePosX[i];
    float y  = particlePosY[i];
    float dx = x - obstacleX;
    float dy = y - obstacleY;

    if (dx*dx + dy*dy < minDist2) {
        particleVelX[i] = obstacleVelX;
        particleVelY[i] = obstacleVelY;
    }

    if (x < minX) { x = minX; particleVelX[i] = 0.0f; }
    if (x > maxX) { x = maxX; particleVelX[i] = 0.0f; }
    if (y < minY) { y = minY; particleVelY[i] = 0.0f; }
    if (y > maxY) { y = maxY; particleVelY[i] = 0.0f; }

    particlePosX[i] = x;
    particlePosY[i] = y;
}

__global__ void updateParticleDensity(
    float* particlePosX, float* particlePosY,
    float* particleDensity,
    float h, float fInvSpacing,
    int fNumX, int fNumY, int numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int   n  = fNumY;
    float hh = h;
    float h1 = fInvSpacing;
    float h2 = 0.5f * h;

    float x = fmaxf(fminf(particlePosX[i], (fNumX - 1) * hh), hh);
    float y = fmaxf(fminf(particlePosY[i], (fNumY - 1) * hh), hh);

    int   x0 = (int)floorf((x - h2) * h1);
    float tx = ((x - h2) - x0 * hh) * h1;
    int   x1 = min(x0 + 1, fNumX - 2);

    int   y0 = (int)floorf((y - h2) * h1);
    float ty = ((y - h2) - y0 * hh) * h1;
    int   y1 = min(y0 + 1, fNumY - 2);

    float sx = 1.0f - tx;
    float sy = 1.0f - ty;

    if (x0 < fNumX && y0 < fNumY) atomicAdd(&particleDensity[x0 * n + y0], sx * sy);
    if (x1 < fNumX && y0 < fNumY) atomicAdd(&particleDensity[x1 * n + y0], tx * sy);
    if (x1 < fNumX && y1 < fNumY) atomicAdd(&particleDensity[x1 * n + y1], tx * ty);
    if (x0 < fNumX && y1 < fNumY) atomicAdd(&particleDensity[x0 * n + y1], sx * ty);
}

__global__ void computeRestDensity(
    float* particleDensity, int* cellType,
    float* outSum, int* outCount, int fNumCells){
    __shared__ float s_sum;
    __shared__ int   s_count;

    if (threadIdx.x == 0) {
        s_sum = 0.0f;
        s_count = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = 0.0f;
    int count = 0;
    if (i < fNumCells && cellType[i] == FLUID_CELL) {
        val = particleDensity[i];
        count = 1;
    }

    atomicAdd(&s_sum, val);
    atomicAdd(&s_count, count);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(outSum, s_sum);
        atomicAdd(outCount, s_count);
    }
}

__global__ void classifyCells(
    int* cellType, float* s,
    float* particlePosX, float* particlePosY,
    float fInvSpacing, int fNumX, int fNumY,
    int fNumCells, int numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < fNumCells)
        cellType[i] = (s[i] == 0.0f) ? SOLID_CELL : AIR_CELL;
}

__global__ void classifyCellsFluid(
    int* cellType,
    float* particlePosX, float* particlePosY,
    float fInvSpacing, int fNumX, int fNumY, int numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    int xi     = max(0, min((int)floorf(particlePosX[i] * fInvSpacing), fNumX - 1));
    int yi     = max(0, min((int)floorf(particlePosY[i] * fInvSpacing), fNumY - 1));
    int cellNr = xi * fNumY + yi;
    atomicCAS(&cellType[cellNr], AIR_CELL, FLUID_CELL);
}

__global__ void savePrevVelocities(
    float* u, float* v,
    float* prevU, float* prevV,
    float* du, float* dv,
    int fNumCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fNumCells) return;

    prevU[i] = u[i];
    prevV[i] = v[i];
    du[i]    = 0.0f;
    dv[i]    = 0.0f;
    u[i]     = 0.0f;
    v[i]     = 0.0f;
}

__global__ void p2gComponent(
    float* particlePosX, float* particlePosY,
    float* particleVelX, float* particleVelY,
    float* u, float* v,
    float* du, float* dv,
    float h, float fInvSpacing,
    int fNumX, int fNumY,
    int numParticles, int component)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int   n     = fNumY;
    float hh    = h;
    float h1    = fInvSpacing;
    float h2    = 0.5f * h;
    float dxOff = (component == 0) ? 0.0f : h2;
    float dyOff = (component == 0) ? h2   : 0.0f;

    float* fld  = (component == 0) ? u  : v;
    float* fldD = (component == 0) ? du : dv;

    float x = fmaxf(fminf(particlePosX[i], (fNumX - 1) * hh), hh);
    float y = fmaxf(fminf(particlePosY[i], (fNumY - 1) * hh), hh);

    int   x0 = min((int)floorf((x - dxOff) * h1), fNumX - 2);
    float tx = ((x - dxOff) - x0 * hh) * h1;
    int   x1 = min(x0 + 1, fNumX - 2);

    int   y0 = min((int)floorf((y - dyOff) * h1), fNumY - 2);
    float ty = ((y - dyOff) - y0 * hh) * h1;
    int   y1 = min(y0 + 1, fNumY - 2);

    float sx = 1.0f - tx;
    float sy = 1.0f - ty;
    float d0 = sx * sy, d1 = tx * sy, d2 = tx * ty, d3 = sx * ty;

    int nr0 = x0 * n + y0;
    int nr1 = x1 * n + y0;
    int nr2 = x1 * n + y1;
    int nr3 = x0 * n + y1;

    float pv = (component == 0) ? particleVelX[i] : particleVelY[i];

    atomicAdd(&fld[nr0],  pv * d0); atomicAdd(&fldD[nr0], d0);
    atomicAdd(&fld[nr1],  pv * d1); atomicAdd(&fldD[nr1], d1);
    atomicAdd(&fld[nr2],  pv * d2); atomicAdd(&fldD[nr2], d2);
    atomicAdd(&fld[nr3],  pv * d3); atomicAdd(&fldD[nr3], d3);
}

__global__ void p2gNormalize(
    float* u, float* v,
    float* du, float* dv,
    int fNumCells, int component)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fNumCells) return;

    if (component == 0) {
        if (du[i] > 0.0f) u[i] /= du[i];
    } else {
        if (dv[i] > 0.0f) v[i] /= dv[i];
    }
}

__global__ void restoreSolidCells(
    float* u, float* v,
    float* prevU, float* prevV,
    int* cellType,
    int fNumX, int fNumY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= fNumX || j >= fNumY) return;

    int n   = fNumY;
    int idx = i * n + j;
    bool solid = (cellType[idx] == SOLID_CELL);

    if (solid || (i > 0 && cellType[(i - 1) * n + j] == SOLID_CELL))
        u[idx] = prevU[idx];
    if (solid || (j > 0 && cellType[i * n + j - 1] == SOLID_CELL))
        v[idx] = prevV[idx];
}

__global__ void g2pComponent(
    float* particlePosX, float* particlePosY,
    float* particleVelX, float* particleVelY,
    float* u, float* v,
    float* prevU, float* prevV,
    int*   cellType,
    float h, float fInvSpacing,
    int fNumX, int fNumY,
    int numParticles, int component, float flipRatio)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int   n      = fNumY;
    float hh     = h;
    float h1     = fInvSpacing;
    float h2     = 0.5f * h;
    float dxOff  = (component == 0) ? 0.0f : h2;
    float dyOff  = (component == 0) ? h2   : 0.0f;
    int   offset = (component == 0) ? n    : 1;

    const float* fld  = (component == 0) ? u     : v;
    const float* pfld = (component == 0) ? prevU : prevV;

    float x = fmaxf(fminf(particlePosX[i], (fNumX - 1) * hh), hh);
    float y = fmaxf(fminf(particlePosY[i], (fNumY - 1) * hh), hh);

    int   x0 = min((int)floorf((x - dxOff) * h1), fNumX - 2);
    float tx = ((x - dxOff) - x0 * hh) * h1;
    int   x1 = min(x0 + 1, fNumX - 2);

    int   y0 = min((int)floorf((y - dyOff) * h1), fNumY - 2);
    float ty = ((y - dyOff) - y0 * hh) * h1;
    int   y1 = min(y0 + 1, fNumY - 2);

    float sx = 1.0f - tx;
    float sy = 1.0f - ty;
    float d0 = sx * sy, d1 = tx * sy, d2 = tx * ty, d3 = sx * ty;

    int nr0 = x0 * n + y0;
    int nr1 = x1 * n + y0;
    int nr2 = x1 * n + y1;
    int nr3 = x0 * n + y1;

    float valid0 = (cellType[nr0] != AIR_CELL || cellType[nr0 - offset] != AIR_CELL) ? 1.0f : 0.0f;
    float valid1 = (cellType[nr1] != AIR_CELL || cellType[nr1 - offset] != AIR_CELL) ? 1.0f : 0.0f;
    float valid2 = (cellType[nr2] != AIR_CELL || cellType[nr2 - offset] != AIR_CELL) ? 1.0f : 0.0f;
    float valid3 = (cellType[nr3] != AIR_CELL || cellType[nr3 - offset] != AIR_CELL) ? 1.0f : 0.0f;

    float d = valid0*d0 + valid1*d1 + valid2*d2 + valid3*d3;
    if (d > 0.0f) {
        float picV  = (valid0*d0*fld[nr0]  + valid1*d1*fld[nr1]  + valid2*d2*fld[nr2]  + valid3*d3*fld[nr3])  / d;
        float corr  = (valid0*d0*(fld[nr0]-pfld[nr0]) + valid1*d1*(fld[nr1]-pfld[nr1])
                     + valid2*d2*(fld[nr2]-pfld[nr2]) + valid3*d3*(fld[nr3]-pfld[nr3])) / d;

        float v_old  = (component == 0) ? particleVelX[i] : particleVelY[i];
        float flipV  = v_old + corr;
        float blended = (1.0f - flipRatio) * picV + flipRatio * flipV;

        if (component == 0) particleVelX[i] = blended;
        else                particleVelY[i] = blended;
    }
}

void transferVelocities(
    float* d_u, float* d_v,
    float* d_du, float* d_dv,
    float* d_prevU, float* d_prevV,
    float* d_s, int* d_cellType,
    float* d_particlePosX, float* d_particlePosY,
    float* d_particleVelX, float* d_particleVelY,
    float  h, float fInvSpacing,
    int    fNumX, int fNumY, int fNumCells,
    int    numParticles,
    bool   toGrid, float flipRatio,
    int threads1D, dim3 threads2D, int blocks1D_cells, dim3 blocks2D_cells) 
{
    int pBlocks = (numParticles + threads1D - 1) / threads1D;

    if (toGrid) {
        savePrevVelocities<<<blocks1D_cells, threads1D>>>(
            d_u, d_v, d_prevU, d_prevV, d_du, d_dv, fNumCells);

        classifyCells<<<blocks1D_cells, threads1D>>>(
            d_cellType, d_s, d_particlePosX, d_particlePosY,
            fInvSpacing, fNumX, fNumY, fNumCells, numParticles);

        classifyCellsFluid<<<pBlocks, threads1D>>>(
            d_cellType, d_particlePosX, d_particlePosY,
            fInvSpacing, fNumX, fNumY, numParticles);

        p2gComponent<<<pBlocks, threads1D>>>(
            d_particlePosX, d_particlePosY,
            d_particleVelX, d_particleVelY,
            d_u, d_v, d_du, d_dv,
            h, fInvSpacing, fNumX, fNumY, numParticles, 0);

        p2gComponent<<<pBlocks, threads1D>>>(
            d_particlePosX, d_particlePosY,
            d_particleVelX, d_particleVelY,
            d_u, d_v, d_du, d_dv,
            h, fInvSpacing, fNumX, fNumY, numParticles, 1);

        p2gNormalize<<<blocks1D_cells, threads1D>>>(
            d_u, d_v, d_du, d_dv, fNumCells, 0);

        p2gNormalize<<<blocks1D_cells, threads1D>>>(
            d_u, d_v, d_du, d_dv, fNumCells, 1);

        restoreSolidCells<<<blocks2D_cells, threads2D>>>(
            d_u, d_v, d_prevU, d_prevV, d_cellType, fNumX, fNumY);
    } else {
        g2pComponent<<<pBlocks, threads1D>>>(
            d_particlePosX, d_particlePosY,
            d_particleVelX, d_particleVelY,
            d_u, d_v, d_prevU, d_prevV, d_cellType,
            h, fInvSpacing, fNumX, fNumY, numParticles, 0, flipRatio);

        g2pComponent<<<pBlocks, threads1D>>>(
            d_particlePosX, d_particlePosY,
            d_particleVelX, d_particleVelY,
            d_u, d_v, d_prevU, d_prevV, d_cellType,
            h, fInvSpacing, fNumX, fNumY, numParticles, 1, flipRatio);
    }
}

__global__ void initPressure(
    float* p, float* prevU, float* prevV,
    float* u, float* v, int fNumCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fNumCells) return;
    p[i]     = 0.0f;
    prevU[i] = u[i];
    prevV[i] = v[i];
}

__global__ void computeDivergence(
    float* u, float* v, float* div,
    float* particleDensity,
    int* cellType, float particleRestDensity, 
    int compensateDrift, int fNumX, int fNumY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= fNumX - 1 || j >= fNumY - 1) return;

    int n = fNumY;
    int center = i * n + j;

    if (cellType[center] != FLUID_CELL) {
        div[center] = 0.0f;
        return;
    }

    int right = (i + 1) * n + j;
    int top   = i * n + j + 1;

    float divergence = u[right] - u[center] + v[top] - v[center];

    if (particleRestDensity > 0.0f && compensateDrift) {
        float compression = particleDensity[center] - particleRestDensity;
        if (compression > 0.0f) divergence -= compression;
    }

    div[center] = divergence;
}

__global__ void jacobiMethod(
    float* p_read, float* p_write,
    float* div, float* s, int* cellType,
    float cp, float overRelaxation, 
    int fNumX, int fNumY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= fNumX - 1 || j >= fNumY - 1) return;

    int n = fNumY;
    int center = i * n + j;

    if (cellType[center] != FLUID_CELL) {
        p_write[center] = 0.0f;
        return;
    }

    int left   = (i - 1) * n + j;
    int right  = (i + 1) * n + j;
    int bottom = i * n + j - 1;
    int top    = i * n + j + 1;

    float sSum = s[left] + s[right] + s[bottom] + s[top];
    if (sSum == 0.0f) {
        p_write[center] = 0.0f;
        return;
    }

    float p_jacobi = (s[left] * p_read[left] + 
                      s[right] * p_read[right] + 
                      s[bottom] * p_read[bottom] + 
                      s[top] * p_read[top] - 
                      (cp * div[center])) / sSum;

    p_write[center] = (1.0f - overRelaxation) * p_read[center] + (overRelaxation * p_jacobi);
}

__global__ void projectVelocity(
    float* u, float* v, float* p, 
    float* s, int* cellType, float cp, 
    int fNumX, int fNumY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= fNumX - 1 || j >= fNumY - 1) return;

    int n = fNumY;
    int center = i * n + j;
    int left   = (i - 1) * n + j;
    int bottom = i * n + j - 1;

    if (s[left] > 0.0f && s[center] > 0.0f && (cellType[center] == FLUID_CELL || cellType[left] == FLUID_CELL)) {
        u[center] -= (p[center] - p[left]) / cp;
    }
    if (s[bottom] > 0.0f && s[center] > 0.0f && (cellType[center] == FLUID_CELL || cellType[bottom] == FLUID_CELL)) {
        v[center] -= (p[center] - p[bottom]) / cp;
    }
}

void solveIncompressibility(
    float* d_u, float* d_v,
    float* d_p, float* &d_p_tmp,
    float* d_div,
    float* d_prevU, float* d_prevV,
    float* d_s, float* d_particleDensity,
    int* d_cellType,
    float  density, float h, float dt,
    float  overRelaxation, bool compensateDrift,
    float  particleRestDensity,
    int    fNumX, int fNumY, int fNumCells,
    int    numIters,
    int threads1D, dim3 threads2D, int blocks1D_cells, dim3 blocks2D_cells)
{
    float cp = density * h / dt;

    initPressure<<<blocks1D_cells, threads1D>>>(
        d_p, d_prevU, d_prevV, d_u, d_v, fNumCells);

    computeDivergence<<<blocks2D_cells, threads2D>>>(
        d_u, d_v, d_div, d_particleDensity, 
        d_cellType, particleRestDensity, 
        compensateDrift ? 1 : 0, fNumX, fNumY);

    for (int iter = 0; iter < numIters; ++iter) {
        jacobiMethod<<<blocks2D_cells, threads2D>>>(
            d_p, d_p_tmp, d_div, d_s, d_cellType,
            cp, overRelaxation, fNumX, fNumY);

        float* tmp = d_p;
        d_p        = d_p_tmp;
        d_p_tmp    = tmp;
    }

    projectVelocity<<<blocks2D_cells, threads2D>>>(
        d_u, d_v, d_p, d_s, d_cellType, 
        cp, fNumX, fNumY);
}

__global__ void updateParticleColors(
    float* particlePosX, float* particlePosY,
    float* particleColorR, float* particleColorG, float* particleColorB,
    float* particleDensity,
    float  particleRestDensity, float fInvSpacing,
    int fNumX, int fNumY, int numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    const float sStep = 0.01f;
    particleColorR[i] = fmaxf(0.0f, fminf(1.0f, particleColorR[i] - sStep));
    particleColorG[i] = fmaxf(0.0f, fminf(1.0f, particleColorG[i] - sStep));
    particleColorB[i] = fmaxf(0.0f, fminf(1.0f, particleColorB[i] + sStep));

    int xi     = max(1, min((int)floorf(particlePosX[i] * fInvSpacing), fNumX - 1));
    int yi     = max(1, min((int)floorf(particlePosY[i] * fInvSpacing), fNumY - 1));
    int cellNr = xi * fNumY + yi;

    if (particleRestDensity > 0.0f) {
        float relDensity = particleDensity[cellNr] / particleRestDensity;
        if (relDensity < 0.7f) {
            particleColorR[i] = 0.8f;
            particleColorG[i] = 0.8f;
            particleColorB[i] = 1.0f;
        }
    }
}

__device__ void setSciColor(float* cellColor, int cellNr, float val, float minVal, float maxVal) {
    val = fmaxf(fminf(val, maxVal - 0.0001f), minVal);
    float d = maxVal - minVal;
    val = (d == 0.0f) ? 0.5f : (val - minVal) / d;
    float m    = 0.25f;
    int   num  = (int)floorf(val / m);
    float sLoc = (val - num * m) / m;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    switch (num) {
        case 0: r = 0.0f; g = sLoc; b = 1.0f;        break;
        case 1: r = 0.0f; g = 1.0f; b = 1.0f - sLoc; break;
        case 2: r = sLoc; g = 1.0f; b = 0.0f;         break;
        default:r = 1.0f; g = 1.0f - sLoc; b = 0.0f; break;
    }
    cellColor[3 * cellNr]     = r;
    cellColor[3 * cellNr + 1] = g;
    cellColor[3 * cellNr + 2] = b;
}

__global__ static void updateCellColors(
    float* cellColor, float* particleDensity,
    int* cellType, float particleRestDensity,
    int fNumCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fNumCells) return;

    if (cellType[i] == SOLID_CELL) {
        cellColor[3 * i]     = 0.5f;
        cellColor[3 * i + 1] = 0.5f;
        cellColor[3 * i + 2] = 0.5f;
    } else if (cellType[i] == FLUID_CELL) {
        float d = particleDensity[i];
        if (particleRestDensity > 0.0f) d /= particleRestDensity;
        setSciColor(cellColor, i, d, 0.0f, 2.0f);
    } else {
        cellColor[3 * i]     = 0.0f;
        cellColor[3 * i + 1] = 0.0f;
        cellColor[3 * i + 2] = 0.0f;
    }
}

void FlipFluid::updateColors() {
    int threads = threads1D;
    int cBlocks = (fNumCells + threads - 1) / threads;
    updateCellColors<<<cBlocks, threads>>>(
        d_cellColor, d_particleDensity,
        d_cellType, particleRestDensity, fNumCells);
}

void FlipFluid::simulate(
    float* d_particlePosX, float* d_particlePosY,
    float* d_particleVelX, float* d_particleVelY,
    float* d_particleColorR, float* d_particleColorG, float* d_particleColorB,
    float* d_particleDensity,
    float* d_u, float* d_v,
    float* d_du, float* d_dv,
    float* d_prevU, float* d_prevV,
    float* &d_p, float* &d_p_tmp,
    float* d_div,
    float* d_s, float* d_cellColor,
    int*   d_cellType,
    int*   d_numCellParticles,
    int*   d_firstCellParticle,
    int*   d_cellParticleIds,
    float  dt, float gravity, float flipRatio,
    int    numPressureIters, int numParticleIters,
    float  overRelaxation, bool compensateDrift,
    bool   separateParticles,
    float  obstacleX, float obstacleY, float obstacleRadius,
    float  obstacleVelX, float obstacleVelY,
    float& particleRestDensity,
    float  density, float h, float fInvSpacing, float pInvSpacing,
    int    fNumX, int fNumY, int fNumCells,
    int    pNumX, int pNumY, int pNumCells,
    int    numParticles, float particleRadius,
    float  colorDiffusionCoeff,
    int    numSubSteps)
{
    if (numSubSteps < 1) numSubSteps = 1;
    float sdt = dt / numSubSteps;

    int pBlocks = (numParticles + threads1D - 1) / threads1D;


    for (int step = 0; step < numSubSteps; ++step) {

        integrateParticles<<<pBlocks, threads1D>>>(
            d_particlePosX, d_particlePosY,
            d_particleVelX, d_particleVelY,
            sdt, gravity, numParticles);

        if (separateParticles) {
            for (int iter = 0; iter < numParticleIters; ++iter) {
                cudaMemset(d_numCellParticles, 0, pNumCells * sizeof(int));

                countParticlesPerCell<<<pBlocks, threads1D>>>(
                    d_particlePosX, d_particlePosY,
                    d_numCellParticles,
                    numParticles, pNumX, pNumY, pInvSpacing);
                
                    prefixSum(d_numCellParticles,d_firstCellParticle,pNumCells, threads1D);

                fillCellParticles<<<pBlocks, threads1D>>>(
                    d_particlePosX, d_particlePosY,
                    d_firstCellParticle, d_cellParticleIds,
                    pInvSpacing, pNumX, pNumY, numParticles);

                pushParticlesApart<<<pBlocks, threads1D>>>(
                    d_particlePosX, d_particlePosY,
                    d_particleColorR, d_particleColorG, d_particleColorB,
                    d_firstCellParticle, d_cellParticleIds,
                    particleRadius, pInvSpacing,
                    pNumX, pNumY, numParticles, colorDiffusionCoeff);
            }
        }

        handleParticleCollisions<<<pBlocks, threads1D>>>(
            d_particlePosX, d_particlePosY,
            d_particleVelX, d_particleVelY,
            obstacleX, obstacleY, obstacleRadius,
            obstacleVelX, obstacleVelY,
            fInvSpacing, particleRadius,
            fNumX, fNumY, numParticles);

        transferVelocities(
            d_u, d_v, d_du, d_dv, d_prevU, d_prevV,
            d_s, d_cellType,
            d_particlePosX, d_particlePosY,
            d_particleVelX, d_particleVelY,
            h, fInvSpacing, fNumX, fNumY, fNumCells,
            numParticles, true, 0.0f,
            threads1D, threads2D, blocks1D_cells, blocks2D_cells);

        cudaMemset(d_particleDensity, 0, fNumCells * sizeof(float));
        updateParticleDensity<<<pBlocks, threads1D>>>(
            d_particlePosX, d_particlePosY,
            d_particleDensity,
            h, fInvSpacing, fNumX, fNumY, numParticles);

        if (particleRestDensity == 0.0f) {
            cudaMemset(d_outSum,   0, sizeof(float));
            cudaMemset(d_outCount, 0, sizeof(int));

            computeRestDensity<<<blocks1D_cells, threads1D>>>(
                d_particleDensity, d_cellType,
                d_outSum, d_outCount, fNumCells);

            float h_sum; int h_count;
            cudaMemcpy(&h_sum,   d_outSum,   sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_count, d_outCount, sizeof(int),   cudaMemcpyDeviceToHost);
            particleRestDensity = (h_count > 0) ? (h_sum / h_count) : 0.0f;
        }

        solveIncompressibility(
            d_u, d_v, d_p, d_p_tmp,
            d_div,
            d_prevU, d_prevV,
            d_s, d_particleDensity, d_cellType,
            density, h, sdt,
            overRelaxation, compensateDrift,
            particleRestDensity,
            fNumX, fNumY, fNumCells, numPressureIters,
            threads1D, threads2D, blocks1D_cells, blocks2D_cells);

        transferVelocities(
            d_u, d_v, d_du, d_dv, d_prevU, d_prevV,
            d_s, d_cellType,
            d_particlePosX, d_particlePosY,
            d_particleVelX, d_particleVelY,
            h, fInvSpacing, fNumX, fNumY, fNumCells,
            numParticles, false, flipRatio,
            threads1D, threads2D, blocks1D_cells, blocks2D_cells);
    }

    updateParticleColors<<<pBlocks, threads1D>>>(
        d_particlePosX, d_particlePosY,
        d_particleColorR, d_particleColorG, d_particleColorB,
        d_particleDensity, particleRestDensity,
        fInvSpacing, fNumX, fNumY, numParticles);

    updateCellColors<<<blocks1D_cells, threads1D>>>(
        d_cellColor, d_particleDensity,
        d_cellType, particleRestDensity, fNumCells);

}

} // namespace flipgpu
