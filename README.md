# FLIP Fluid Simulator (CPU & GPU CUDA Port)

## Build

```bash
# CPU Version
make -C src/cpu clean && make -C src/cpu

# GPU Version
make -C src/cuda clean && make -C src/cuda
```

## Run

On WSL, ensure that you have WSLg or an X server configured. 

```bash
# CPU Version
./src/cpu/flip --no-vsync

# GPU Version
./src/cuda/flip-gpu --no-vsync
```

## Controls
- **Mouse Left Drag**: Move obstacle
- **Space / P**: Pause/Resume
- **G**: Toggle Grid
- **R**: Reset
- **Q / ESC**: Quit
