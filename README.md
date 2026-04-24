# CFD-ML Surrogate for Heat Sink Thermal Prediction

> **Neural network surrogate model that replaces a 15-minute CFD simulation with a 40-millisecond inference,a 22,000× speedup for heat sink thermal design.**

Paper:
**"Machine Learning Applied for Instant Predictions of Spatial Temperature Variations in Heat Sinks for Computer Chip Cooling"**
A. Jha, R. Thimmaiah, I. Perez-Raya — Rochester Institute of Technology
*Accepted to the ASME 2026 Fluids Engineering Division Summer Meeting (FEDSM2026-184041), Bellevue, WA, July 2026.*

---

## Why this matters

Thermal design of electronic cooling systems is bottlenecked by CFD. A single OpenFOAM simulation of a pin-fin heat sink takes ~15 minutes on an 8-core CPU, and exploring even a modest design space of 1,000 geometries takes ~250 CPU-hours. This work trains a PyTorch neural network on just 25 CFD simulations.

Once trained, the surrogate predicts the full 3D temperature field T(x, y, z) for any fin geometry in the training range in **40 ms**, enabling real-time design iteration and gradient-based optimization that is infeasible with direct CFD.

## Results

| Metric | Single-fin study | 6×6 heat sink study |
|---|---|---|
| Training data | 25 analytical cases | 25 CFD simulations (~1.9M mesh points) |
| Inputs | x, y, z, L | x, y, z, L, w |
| Architecture | 4 → 128 → 256 → 128 → 64 → 1 | Same |
| R² on held-out test set | **0.999998** | **0.999997** |
| MAE | 0.07 K | 0.021 K |
| RMSE | 0.09 K | 0.029 K |
| Training time | ~19 s | ~2 h (GPU) |
| Inference time | 0.02 s | 0.04 s |
| Speedup vs CFD | — | **~22,500×** |

Predicted-vs-actual temperature on the test set (6×6 heat sink, 383,138 points):

- Points lie on the identity line across the full 343 K–408 K range
- No overfitting: training and validation losses track together throughout training

## Method

1. **Parametric geometry generation** — 25 pin-fin heat sink geometries swept in SOLIDWORKS (fin length 10–20 mm, fin width 1.5–5.0 mm)
2. **CFD ground truth** — conjugate heat transfer solved in OpenFOAM using `chtMultiRegionFoam` with convective BCs (h = 125 W/m²K, T∞ = 40 °C) and 50 W chip heat generation
3. **Data pipeline** — ~1.9M cell-centered (x, y, z, T) samples pooled across geometries, z-score normalized
4. **Surrogate model** — fully-connected feedforward network in PyTorch, Adam optimizer, MSE loss, early stopping with patience
5. **Validation** — 64/16/20 train/val/test split, verified against analytical fin-theory solution (< 1% error)

## Repository structure

Open either notebook directly in Colab using the badge at the top of the file.

## Tech stack

PyTorch · NumPy · pandas · scikit-learn · matplotlib · OpenFOAM (`chtMultiRegionFoam`) · SOLIDWORKS

## Related work / context

This project sits in the broader space of **physics-informed and data-driven surrogates for CFD**, an active area of research and industry practice:
- NVIDIA Modulus / PhysicsNeMo — neural PDE solvers and surrogates
- DeepMind GraphCast & FourCastNet — learned weather/fluids surrogates
- Fourier Neural Operators (FNO) and DeepONet for PDE operators

A natural extension of this work is a **Physics-Informed Neural Network (PINN)** that embeds the steady-state heat equation and convective boundary conditions directly into the loss function — planned follow-up.

## Citation

```bibtex
@inproceedings{jha2026heatsink,
  author    = {Jha, Ayush and Thimmaiah, Reetu and Perez-Raya, Isaac},
  title     = {Machine Learning Applied for Instant Predictions of Spatial Temperature
               Variations in Heat Sinks for Computer Chip Cooling},
  booktitle = {Proceedings of the ASME 2026 Fluids Engineering Division Summer Meeting},
  year      = {2026},
  number    = {FEDSM2026-184041},
  address   = {Bellevue, Washington, USA},
  month     = {July},
  note      = {Accepted}
}
```

## Authors

- **Reetu Thimmaiah** — Rochester Institute of Technology — [GitHub](https://github.com/reetu95)
- Ayush Jha — Rochester Institute of Technology
- Isaac Perez-Raya — Rochester Institute of Technology
