<div align="center">

[![](https://zenodo.org/badge/DOI/10.5281/zenodo.17715490.svg)](https://doi.org/10.5281/zenodo.17715490)

</div>

# WaterFlow-Thermodynamic-AI

## 💧 Overview: Reversible AI Trained with Thermodynamic Loss

This repository contains a prototype implementation of a Normalizing Flow model, dubbed **WaterFlow**, which is trained using a **Thermodynamic Loss function ($\mathcal{F}$) and Beta-Annealing**.

The goal is to create an AI system that seeks **Thermodynamic Equilibrium**, minimizing the Helmholtz Free Energy ($\mathcal{F}$) defined as:

$$\mathcal{F} = \beta \cdot E(x) - \lambda \cdot S_{regulations}$$

Where:
* **$\beta$ (Inverse Temperature):** Controls the balance between performance and stability. Annealed from 0 to 1 during training.
* **$E(x)$ (Task Energy):** The Negative Log-Likelihood (NLL) of the data. Lower $E$ means better performance.
* **$S_{regulations}$ (Entropy):** The Shannon Entropy of the model's weights, encouraging structural simplicity and non-randomness.

## ⚙️ Model Architecture

The `WaterFlow` model is a Generative AI model built upon **Reversible Computing** principles:

1.  **Reversible Block:** Uses the **Affine Coupling Layer** from Normalizing Flows, guaranteeing perfect reversibility (non-disipative flow).
2.  **Stacked Flow:** Stacks 16 `AffineCoupling` layers.

## 🚀 How to Run

### Requirements

This project requires PyTorch and standard libraries:
```bash
pip install torch torchvision numpy
```
###Training
**1. Save the Python code above as waterflow_model.py**

**2.Run the script. It will automatically download the MNIST dataset and begin training on a CUDA device (GPU) if available.**

```bash
python waterflow_model.py
```
The output log will show the progressive decrease of $E(x)$ (Task Energy) as $\beta$ (Inverse Temperature) increases, demonstrating the model's convergence toward a stable thermodynamic state.
