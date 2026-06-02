

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

struct TimingStats {
    double t1_integrate = 0.0;
    double t2_pushApart = 0.0;
    double t3_collisions = 0.0;
    double t4_p2g = 0.0;
    double t5_density = 0.0;
    double t6_pressure = 0.0;
    double t7_g2p = 0.0;
    double t8_colors = 0.0;
    double t9_render = 0.0;
    double t10_d2h = 0.0;
};

class FlipFluid {
public:
    TimingStats lastFrameStats;

    cudaEvent_t start_t1, stop_t1;
    cudaEvent_t start_t2, stop_t2;
    cudaEvent_t start_t3, stop_t3;
    cudaEvent_t start_t4, stop_t4;
    cudaEvent_t start_t5, stop_t5;
    cudaEvent_t start_t6, stop_t6;
    cudaEvent_t start_t7, stop_t7;
    cudaEvent_t start_t8, stop_t8;

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
