

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

namespace flipgpu {

constexpr int U_FIELD = 0;
constexpr int V_FIELD = 1;

constexpr int FLUID_CELL = 0;
constexpr int AIR_CELL   = 1;
constexpr int SOLID_CELL = 2;

#include <vector>

class FlipFluid {
public:
    int threads1D;
    dim3 threads2D;
    
    int blocks1D_cells;
    dim3 blocks2D_cells;
    int blocks1D_pCells;

    float density;
    int   fNumX;
    int   fNumY;
    float h;
    float fInvSpacing;
    int   fNumCells;

    std::vector<float> u, v, du, dv, prevU, prevV, p, s;
    std::vector<int>   cellType;
    std::vector<float> cellColor;        // 3 * fNumCells

    int   maxParticles;
    int   numParticles = 0;
    float particleRestDensity = 0.0f;
    float particleRadius;
    float pInvSpacing;
    int   pNumX;
    int   pNumY;
    int   pNumCells;
    float colorDiffusionCoeff = 0.001f;

    std::vector<float> particlePosX, particlePosY;
    std::vector<float> particleVelX, particleVelY;
    std::vector<float> particleColorR, particleColorG, particleColorB;
    std::vector<float> particleDensity;

    std::vector<int> numCellParticles;
    std::vector<int> firstCellParticle;
    std::vector<int> cellParticleIds;

    float* d_u;          float* d_v;
    float* d_du;         float* d_dv;
    float* d_prevU;      float* d_prevV;
    float* d_p;          float* d_p_tmp;
    float* d_s;          float* d_div;
    float* d_cellColor;
    int* d_cellType;

    float* d_particlePosX;   float* d_particlePosY;
    float* d_particleVelX;   float* d_particleVelY;
    float* d_particleColorR; float* d_particleColorG; float* d_particleColorB;
    float* d_particleDensity;

    float* d_outSum; int* d_outCount;

    int* d_numCellParticles;
    int* d_firstCellParticle;
    int* d_cellParticleIds;

    FlipFluid(float density_, float width, float height, float spacing, 
              float particle_radius, int max_particles, 
              int threads_1d = 256, dim3 threads_2d = dim3(16, 16));
    ~FlipFluid();

    void uploadToGPU();

    void updateColors();
    void simulate(
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
    int    numSubSteps);
};

} // namespace flipgpu
