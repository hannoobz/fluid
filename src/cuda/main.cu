// Window + renderer + main loop for the C++ FLIP demo.
// Simulation runs on the GPU
//
// Inputs:
//   left mouse drag   move/release the obstacle
//   SPACE / P         pause-resume
//   G                 toggle grid
//   R                 reset scene
//   Q / Esc           quit

#include "flip_fluid.h"
#include "ui.h"

#ifdef __APPLE__
#  define GL_SILENCE_DEPRECATION
#endif
#include <GLFW/glfw3.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace flipgpu;

// --------------------------- scene / config ---------------------------------
struct Scene {
    float gravity         = -9.81f;
    float dt              = 1.0f / 60.0f;
    float flipRatio       = 0.9f;
    int   numPressureIters= 50;
    int   numParticleIters= 2;
    long  frameNr         = 0;
    float overRelaxation  = 1.0f;
    bool  compensateDrift = true;
    bool  separateParticles = true;
    float obstacleX       = 0.0f;
    float obstacleY       = 0.0f;
    float obstacleRadius  = 0.15f;
    bool  paused          = true;
    bool  showObstacle    = true;
    float obstacleVelX    = 0.0f;
    float obstacleVelY    = 0.0f;
    bool  showParticles   = true;
    bool  showGrid        = false;
    int   resolution      = 100;   // grid cells along the tank height
    int   numSubSteps     = 1;     // CFL substeps — auto-scaled with res
    FlipFluid* fluid      = nullptr;
};

static Scene scene;

constexpr int CANVAS_W = 900;
constexpr int CANVAS_H = 700;
constexpr float simHeight = 3.0f;
constexpr float cScale = float(CANVAS_H) / simHeight;
constexpr float simWidth = float(CANVAS_W) / cScale;

constexpr int THREADS_1D = 256;
const dim3 THREADS_2D(16, 16);

// --------------------------- scene helpers ----------------------------------
__global__ void carveObstacle(
    float* d_u, float* d_v, float* d_s,
    float obsX, float obsY, float obsR,
    float obsVx, float obsVy,
    int fNumX, int fNumY, float h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i < fNumX - 2 && j >= 1 && j < fNumY - 2) {
        int n = fNumY;
        int idx = i * n + j;
        
        float dx = (i + 0.5f) * h - obsX;
        float dy = (j + 0.5f) * h - obsY;
        
        d_s[idx] = 1.0f;
        
        if (dx * dx + dy * dy < obsR * obsR) {
            d_s[idx] = 0.0f; 
            d_u[idx]     = obsVx;
            d_u[idx + n] = obsVx;
            d_v[idx]     = obsVy;
            d_v[idx + 1] = obsVy;
        }
    }
}

static void setObstacle(float x, float y, bool reset) {
    float vx = 0.0f, vy = 0.0f;
    if (!reset) {
        vx = (x - scene.obstacleX) / scene.dt;
        vy = (y - scene.obstacleY) / scene.dt;
    }
    scene.obstacleX = x;
    scene.obstacleY = y;

    carveObstacle<<<scene.fluid->blocks2D_cells, scene.fluid->threads2D>>>(
        scene.fluid->d_u, scene.fluid->d_v, scene.fluid->d_s,
        x, y, scene.obstacleRadius, vx, vy,
        scene.fluid->fNumX, scene.fluid->fNumY, scene.fluid->h
    );

    scene.showObstacle  = true;
    scene.obstacleVelX  = vx;
    scene.obstacleVelY  = vy;
}

static void seedParticles(FlipFluid& f, int numX, int numY,
                          float h, float r, float dx, float dy)
{
    for (int i = 0; i < numX; ++i) {
        for (int j = 0; j < numY; ++j) {
            int pid = i * numY + j;
            float offset = (j % 2 == 0) ? 0.0f : r;
            f.particlePosX[pid] = h + r + dx * i + offset;
            f.particlePosY[pid] = h + r + dy * j;
        }
    }
}

static void setupTank(FlipFluid& f) {
    int n = f.fNumY;
    for (int i = 0; i < f.fNumX; ++i) {
        for (int j = 0; j < f.fNumY; ++j) {
            float sVal = 1.0f;
            if (i == 0 || i == f.fNumX - 1 || j == 0) sVal = 0.0f;
            f.s[i * n + j] = sVal;
        }
    }
}

static void setupScene() {
	scene.obstacleRadius = 0.15f;
	scene.overRelaxation = 1.0f;
	scene.dt = 1.0f / 60.0f;
	scene.numParticleIters = 2;

	int res = scene.resolution;

	if (res <= 100) scene.numSubSteps = 1;
	else if (res <= 140) scene.numSubSteps = 2;
	else if (res <= 180) scene.numSubSteps = 3;
	else scene.numSubSteps = 4;
	scene.numPressureIters = 50 + std::max(0, (res - 100)) / 2;
	float tankHeight = 1.0f * simHeight;
	float tankWidth = 1.0f * simWidth;
	float h = tankHeight / res;
	float density = 1000.0f;

	float relWaterHeight = 0.8f;
	float relWaterWidth = 0.6f;

	float r = 0.3f * h;
	float dx = 2.0f * r;
	float dy = std::sqrt(3.0f) / 2.0f * dx;

	int numX = int(std::floor((relWaterWidth * tankWidth - 2.0f * h - 2.0f * r) / dx));
	int numY = int(std::floor((relWaterHeight * tankHeight - 2.0f * h - 2.0f * r) / dy));
	if (numX < 1) numX = 1;
	if (numY < 1) numY = 1;
	int maxParticles = numX * numY;

	delete scene.fluid;
	scene.fluid = new FlipFluid(density, tankWidth, tankHeight, h, r, maxParticles, THREADS_1D, THREADS_2D);

	FlipFluid &f = *scene.fluid;
	f.numParticles = numX * numY;
	f.particleRestDensity = 0.0f;

	seedParticles(f, numX, numY, h, r, dx, dy);
	setupTank(f);
	setObstacle(3.0f, 2.0f, true);
	scene.frameNr = 0;
	scene.fluid->uploadToGPU();
}

// --------------------------- GLFW input state -------------------------------

static int  s_winW = CANVAS_W, s_winH = CANVAS_H;
static int  s_fbW  = CANVAS_W, s_fbH  = CANVAS_H;
static bool s_mouseDown = false;
static bool s_mousePressedEdge = false;
static bool s_mouseReleasedEdge = false;
static double s_mouseX = 0.0, s_mouseY = 0.0;   // window coords, y-down

static void keyCallback(GLFWwindow* win, int key, int /*scancode*/, int action, int /*mods*/) {
    if (action != GLFW_PRESS) return;
    switch (key) {
        case GLFW_KEY_SPACE:
        case GLFW_KEY_P:
            scene.paused = !scene.paused;
            break;
        case GLFW_KEY_G:
            scene.showGrid = !scene.showGrid;
            break;
        case GLFW_KEY_R:
            setupScene();
            break;
        case GLFW_KEY_Q:
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(win, GLFW_TRUE);
            break;
        default: break;
    }
}

static void mouseButtonCallback(GLFWwindow* /*win*/, int button, int action, int /*mods*/) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    if (action == GLFW_PRESS) {
        s_mouseDown = true;
        s_mousePressedEdge = true;
    } else if (action == GLFW_RELEASE) {
        s_mouseDown = false;
        s_mouseReleasedEdge = true;
    }
}

static void cursorPosCallback(GLFWwindow* /*win*/, double xpos, double ypos) {
    s_mouseX = xpos;
    s_mouseY = ypos;
}

static void framebufferSizeCallback(GLFWwindow* /*win*/, int w, int h) {
    s_fbW = w;
    s_fbH = h;
}

static void windowSizeCallback(GLFWwindow* /*win*/, int w, int h) {
    s_winW = w;
    s_winH = h;
}

// --------------------------- rendering --------------------------------------

static void setProjection(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, simWidth, 0.0, simHeight, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

static void drawGrid(FlipFluid& f) {
    cudaMemcpy(f.cellColor.data(), f.d_cellColor, 3 * f.fNumCells * sizeof(float), cudaMemcpyDeviceToHost);

    float h = f.h;
    glBegin(GL_QUADS);
    for (int i = 0; i < f.fNumX; ++i) {
        for (int j = 0; j < f.fNumY; ++j) {
            int idx = i * f.fNumY + j;
            float r = f.cellColor[3 * idx + 0];
            float g = f.cellColor[3 * idx + 1];
            float b = f.cellColor[3 * idx + 2];
            float x0 = i * h, y0 = j * h;
            float x1 = x0 + h, y1 = y0 + h;
            glColor3f(r, g, b);
            glVertex2f(x0, y0);
            glVertex2f(x1, y0);
            glVertex2f(x1, y1);
            glVertex2f(x0, y1);
        }
    }
    glEnd();
}

static void drawParticles(FlipFluid &f, int viewportH) {
    cudaMemcpy(f.particlePosX.data(),   f.d_particlePosX,   f.numParticles * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(f.particlePosY.data(),   f.d_particlePosY,   f.numParticles * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(f.particleColorR.data(), f.d_particleColorR, f.numParticles * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(f.particleColorG.data(), f.d_particleColorG, f.numParticles * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(f.particleColorB.data(), f.d_particleColorB, f.numParticles * sizeof(float), cudaMemcpyDeviceToHost);

    float pxPerSimUnit = float(viewportH) / simHeight;
    float diameterPx = 2.0f * f.particleRadius * pxPerSimUnit;
    if (diameterPx < 1.0f) diameterPx = 1.0f;
    glPointSize(diameterPx);
    glBegin(GL_POINTS);
    for (int i = 0; i < f.numParticles; ++i) {
        glColor3f(f.particleColorR[i], f.particleColorG[i], f.particleColorB[i]);
        glVertex2f(f.particlePosX[i], f.particlePosY[i]);
    }
    glEnd();
}

static void drawObstacle(const FlipFluid &f, float ox, float oy, float orad) {
	const int N = 48;
	float drawR = orad + f.particleRadius;
	glColor3f(1.0f, 0.0f, 0.0f);
	glBegin(GL_TRIANGLE_FAN);
	glVertex2f(ox, oy);
	for (int i = 0; i <= N; ++i) {
		float a = (float)i / N * 2.0f * (float)M_PI;
		glVertex2f(ox + drawR * std::cos(a), oy + drawR * std::sin(a));
	}
	glEnd();
}

// --------------------------- main -------------------------------------------

int main(int argc, char** argv) {
    bool noVsync = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-vsync") == 0) noVsync = true;
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            std::printf("Usage: %s [--no-vsync]\n", argv[0]);
            std::printf("Controls: LMB=move obstacle, SPACE/P=pause, G=grid, R=reset, Q/Esc=quit\n");
            return 0;
        }
    }

    std::printf("[flip-cu] starting\n");
    setupScene();
    scene.paused = true;

    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    // Request a legacy (compatibility) OpenGL context — macOS needs this
    // for glBegin/glEnd immediate-mode calls.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    GLFWwindow* win = glfwCreateWindow(CANVAS_W, CANVAS_H,
                                        "FLIP Fluid (C++ GPU sim)", nullptr, nullptr);
    if (!win) {
        std::fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(noVsync ? 0 : 1);

    glfwSetKeyCallback(win, keyCallback);
    glfwSetMouseButtonCallback(win, mouseButtonCallback);
    glfwSetCursorPosCallback(win, cursorPosCallback);
    glfwSetFramebufferSizeCallback(win, framebufferSizeCallback);
    glfwSetWindowSizeCallback(win, windowSizeCallback);

    glfwGetWindowSize(win, &s_winW, &s_winH);
    glfwGetFramebufferSize(win, &s_fbW, &s_fbH);

    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    bool mouseDownPrev = false;
    float mouseSimX = 0.0f, mouseSimY = 0.0f;
    bool dragOwnedByUI = false;

    auto fpsT0 = std::chrono::steady_clock::now();
    int  fpsFrames = 0;
    double lastFps = 0.0;
    bool gravityOn = (scene.gravity != 0.0f);

    while (!glfwWindowShouldClose(win)) {
        s_mousePressedEdge = false;
        s_mouseReleasedEdge = false;

        glfwPollEvents();

        int mousePxX = (int)s_mouseX;
        int mousePxY = (int)s_mouseY;
        mouseSimX = float(s_mouseX) / s_winW  * simWidth;
        mouseSimY = (1.0f - float(s_mouseY) / s_winH) * simHeight;

        const int kPanelX = 10, kPanelY = 10, kPanelW = 160, kPanelH = 250;
        bool mouseOnPanel =
            (mousePxX >= kPanelX && mousePxX < kPanelX + kPanelW &&
             mousePxY >= kPanelY && mousePxY < kPanelY + kPanelH);
        if (s_mousePressedEdge && mouseOnPanel) dragOwnedByUI = true;
        if (!s_mouseDown) dragOwnedByUI = false;

        FlipFluid& f = *scene.fluid;

        // ----- obstacle drag -----
        if (s_mouseDown && !dragOwnedByUI) {
            if (!mouseDownPrev) {
                setObstacle(mouseSimX, mouseSimY, true);
                scene.paused = false;
            } else {
                setObstacle(mouseSimX, mouseSimY, false);
            }
            mouseDownPrev = true;
        } else {
            if (mouseDownPrev) {
                scene.obstacleVelX = 0.0f;
                scene.obstacleVelY = 0.0f;
                setObstacle(scene.obstacleX, scene.obstacleY, true); 
            }
            mouseDownPrev = false;
        }

        // ----- step -----
		if (!scene.paused) {
            f.simulate(
            f.d_particlePosX, f.d_particlePosY,
            f.d_particleVelX, f.d_particleVelY,
            f.d_particleColorR, f.d_particleColorG, f.d_particleColorB,
            f.d_particleDensity,
            f.d_u, f.d_v,
            f.d_du, f.d_dv,
            f.d_prevU, f.d_prevV,
            f.d_p, f.d_p_tmp,
            f.d_div,
            f.d_s, f.d_cellColor,
            f.d_cellType,
            f.d_numCellParticles,
            f.d_firstCellParticle,
            f.d_cellParticleIds,
            scene.dt, scene.gravity, scene.flipRatio,
            scene.numPressureIters, scene.numParticleIters,
            scene.overRelaxation, scene.compensateDrift,
            scene.separateParticles,
            scene.obstacleX, scene.obstacleY, scene.obstacleRadius,
            scene.obstacleVelX, scene.obstacleVelY,
            f.particleRestDensity,
            f.density, f.h, f.fInvSpacing, f.pInvSpacing,
            f.fNumX, f.fNumY, f.fNumCells,
            f.pNumX, f.pNumY, f.pNumCells,
            f.numParticles, f.particleRadius,
            f.colorDiffusionCoeff,
            scene.numSubSteps);
        scene.frameNr += 1;
        } else {
            if (scene.frameNr == 0) {
                f.updateColors();
            }
        }


        // ----- draw sim -----
        glClear(GL_COLOR_BUFFER_BIT);
        setProjection(s_fbW, s_fbH);

        if (scene.showGrid)      drawGrid(f);
        if (scene.showParticles) drawParticles(f, s_fbH);
        if (scene.showObstacle)  drawObstacle(f, scene.obstacleX,
                                              scene.obstacleY,
                                              scene.obstacleRadius);

        // ----- draw UI overlay -----
        flipgpu_ui::setProjectionToPixels(s_winW, s_winH);

        flipgpu_ui::Input uin;
        uin.screenW = s_winW;
        uin.screenH = s_winH;
        uin.mouseX  = mousePxX;
        uin.mouseY  = mousePxY;
        uin.mouseDown    = s_mouseDown;
        uin.mousePressed = s_mousePressedEdge && mouseOnPanel;
        uin.mouseReleased = s_mouseReleasedEdge;
        flipgpu_ui::begin(uin);

        flipgpu_ui::beginPanel(kPanelX, kPanelY, kPanelW, kPanelH, "Controls");
        flipgpu_ui::text("FPS: %.1f", lastFps);
        flipgpu_ui::text("Particles: %d", f.numParticles);
        flipgpu_ui::text("Frame: %ld", scene.frameNr);
        flipgpu_ui::checkbox("Particles",          &scene.showParticles);
        flipgpu_ui::checkbox("Grid",               &scene.showGrid);
        flipgpu_ui::checkbox("Compensate Drift",   &scene.compensateDrift);
        flipgpu_ui::checkbox("Separate Particles", &scene.separateParticles);
        if (flipgpu_ui::checkbox("Gravity", &gravityOn)) {
            scene.gravity = gravityOn ? -9.81f : 0.0f;
        }
        flipgpu_ui::slider("PIC <-> FLIP", &scene.flipRatio, 0.0f, 1.0f);
        float resFloat = (float)scene.resolution;
        flipgpu_ui::slider("Grid Res", &resFloat, 30.0f, 200.0f);
        int newRes = (int)(resFloat + 0.5f);
        if (newRes != scene.resolution) {
            scene.resolution = newRes;
            setupScene();
        }
        flipgpu_ui::checkbox("Pause", &scene.paused);
        if (flipgpu_ui::button("Reset")) setupScene();
        flipgpu_ui::endPanel();

        flipgpu_ui::restoreProjection();

        glfwSwapBuffers(win);

        // ----- fps -----
		fpsFrames += 1;
		auto now = std::chrono::steady_clock::now();
		double elapsed = std::chrono::duration<double>(now - fpsT0).count();
		if (elapsed >= 0.5) {
			lastFps = fpsFrames / elapsed;
			
			cudaError_t err = cudaGetLastError();
			const char* gpuStatus = (err == cudaSuccess) ? "GPU_OK" : cudaGetErrorString(err);
			
			std::printf("[flip-cu] %5.1f FPS | Frame: %6ld | GPU Status: %-35s \n", 
						lastFps, scene.frameNr, gpuStatus);
			std::fflush(stdout);
			
			fpsT0 = now;
			fpsFrames = 0;
		}
	}

    glfwDestroyWindow(win);
    glfwTerminate();
    delete scene.fluid;
    return 0;
}
