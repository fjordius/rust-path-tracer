# Rust Path Tracer in a Single File

This repository showcases a self-contained Rust implementation of a Monte Carlo path tracer. The renderer casts rays through a stochastic camera model, simulates light interactions with a variety of materials, and produces a photorealistic PPM image without relying on external crates.

## Highlights

- **Single source file**: The complete implementation lives in [`main.rs`](./main.rs).
- **Physically based materials**: Diffuse, metallic (with configurable fuzz), and dielectric glass.
- **Custom RNG**: A lightweight XorShift generator keeps the project dependency-free.
- **Depth of field & anti-aliasing**: Thin-lens camera and multi-sample integration for smooth imagery.

## Building and Running

Install the Rust toolchain (Rust 1.70+ recommended), then execute the following from the project directory:

```powershell
rustc main.rs
./main.exe
```

The renderer writes `output.ppm`. On Windows, launch it directly with:

```powershell
start output.ppm
```

## Converting the Output

To convert the PPM to PNG using ImageMagick:

```powershell
magick convert output.ppm output.png
```

## Implementation Overview

1. `random_scene` populates the world with a ground plane and randomized spheres.@rust_path_tracer/main.rs#66-119
2. The main render loop jitters camera rays per pixel to accumulate color samples.@rust_path_tracer/main.rs#31-43
3. `Material::scatter` governs reflection, refraction, and absorption for each surface interaction.@rust_path_tracer/main.rs#399-452
4. `Camera` implements a thin-lens model to simulate focus and depth of field.@rust_path_tracer/main.rs#502-555
5. `write_color` applies gamma correction before writing to the PPM file.@rust_path_tracer/main.rs#121-132

## License

This project is released under the MIT License. Contributions and render showcases are always welcome.
