# Machine Learning for Gravitational Waveforms

This repository contains the source code for the `mlgw_bns_HOM` project, which aims to accelerate the generation of gravitational waveforms from binary neutron star (BNS) mergers using machine learning techniques.

Currently, the training data is primarily generated using [TEOBResumS](https://teobresums.bitbucket.io/), a state-of-the-art waveform model. However, support for alternative waveform models can be added by modifying the [`higher_order_modes.py`](src/higher_order_modes.py) script.

## 📦 Installation

### 1. Install TEOBResumS (Required)

Waveform generation relies on the [TEOBResumS](https://teobresums.bitbucket.io/) library. You must install it before using this package.  
Follow the installation guide here:  
👉 [TEOBResumS Installation Guide](https://bitbucket.org/teobresums/teobresums/src/GIOTTO/)

### 2. Install Poetry (Recommended)

This project uses [Poetry](https://python-poetry.org/) for dependency management and virtual environments.

To install Poetry:

```bash
pip install poetry
```

### 3. Set up the environment and install dependencies

Once Poetry is installed, run:

```bash
poetry install
```
This will create a virtual environment and install all necessary dependencies.

### 4. Run the code

To execute the waveform generation script:

```bash
poetry run python generate_model.py
```

## 📚 Publications and Academic Work

This repository is associated with the following academic work:

- **Master's Thesis:**  
  *Machine Learning for Gravitational Waveforms*  
  [Prasoon Pandey], [Friedrich Schiller University Jena], 2025  
  [📄 View Thesis](https://github.com/ppandey10/msc_thesis)

- **Research Paper:**  
  A related research paper is currently in preparation and will be linked here once available.






