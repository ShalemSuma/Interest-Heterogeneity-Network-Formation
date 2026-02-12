# Interest Heterogeneity in Network Formation on Large Online Platforms

This repository contains the code and analysis for the research paper
*"Interest Heterogeneity in Network Formation on Large Online Platforms"*,
which investigates how information heterogeneity and strategic interests
shape network formation, testing the "Law of the Few" hypothesis using
YouTube subscription data.

## Research Question

> How does interest heterogeneity among agents influence network formation
> patterns, and under what conditions do multiple specialised hubs emerge
> rather than a single core-periphery structure?

## Key Findings

- The theoretical model (Galeotti & Goyal, 2010) predicts a small number of
  information hubs serving a large periphery — the "Law of the Few".
- Introducing **heterogeneous interest vectors** leads to the emergence of
  **multiple interest-specific hubs**, each serving a community of nodes with
  similar interests.
- Empirical analysis of the **YouTube subscription network** confirms that
  core nodes (top 3.3% by centrality) exhibit significantly different interest
  profiles from peripheral nodes.
- Logistic regression identifies **interest mean**, **interest Gini coefficient**,
  and **interest kurtosis** as the strongest predictors of hub status.

## Repository Structure

```
├── src/                          # Reusable Python modules
│   ├── models/
│   │   ├── law_of_few.py         # Galeotti & Goyal (2010) base model
│   │   ├── interest_law_of_few.py# Interest-based extension
│   │   ├── network_formation.py  # Bala & Goyal (2000) simulation
│   │   └── hub_prediction.py     # ML hub prediction (RF + GBM)
│   ├── collection/
│   │   └── youtube_collector.py  # YouTube Data API v3 collector
│   ├── analysis/
│   │   ├── interest_vectors.py   # TF-IDF pipeline, 703→30 dim reduction
│   │   ├── network_analysis.py   # Centrality calculations, core detection
│   │   └── statistical_analysis.py # Logistic regression pipelines
│   ├── visualization/
│   │   ├── network_plots.py      # Network layout visualisations
│   │   ├── simulation_plots.py   # Simulation evolution plots
│   │   └── comparison_plots.py   # Real vs simulation comparison
│   └── utils/
│       └── config.py             # Env-var loading, path configuration
├── notebooks/                    # Analysis notebooks (run in order)
│   ├── 01_theoretical_model.ipynb
│   ├── 02_network_formation_simulation.ipynb
│   ├── 03_hub_prediction.ipynb
│   ├── 04_youtube_data_collection.ipynb
│   ├── 05_interest_vector_pipeline.ipynb
│   ├── 06_network_analysis.ipynb
│   └── 07_statistical_analysis.ipynb
├── data/
│   ├── raw/                      # Full datasets (not tracked by git)
│   ├── processed/                # Intermediate outputs
│   └── sample/                   # Small samples for testing
├── results/
│   ├── figures/                  # All visualisations
│   └── tables/                   # Feature importance, analysis CSVs
└── docs/
    └── paper.pdf                 # Research paper
```

## Installation

```bash
# Clone the repository
git clone https://github.com/<username>/interest-heterogeneity-network-formation.git
cd interest-heterogeneity-network-formation

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Setup

For YouTube data collection, create a `.env` file from the template:

```bash
cp .env.example .env
# Edit .env with your YouTube Data API credentials
```

## Quick Start

The notebooks are numbered in suggested execution order:

| # | Notebook | Description | Requires Data? |
|---|----------|-------------|:--------------:|
| 01 | Theoretical Model | Law of the Few base simulation | No |
| 02 | Network Formation | Bala & Goyal strategic network formation | No |
| 03 | Hub Prediction | ML pipeline for predicting hub emergence | No |
| 04 | YouTube Collection | Data collection via YouTube API | API keys |
| 05 | Interest Vectors | TF-IDF interest vector generation | Video data |
| 06 | Network Analysis | Centrality and core-node detection | Network data |
| 07 | Statistical Analysis | Logistic regression on real + sim data | No (synthetic) |

Notebooks 01-03 and 07 can be run immediately with no external data.
Notebooks 04-06 require the YouTube dataset (see `data/raw/README.md`).

## Models

### Law of the Few (Galeotti & Goyal, 2010)
Players choose between acquiring information and linking to others.
Equilibrium produces a small "core" of informed hubs.

### Interest-Based Extension
Adds multidimensional interest vectors so link costs depend on interest
similarity, producing multiple specialised hubs.

### Noncooperative Network Formation (Bala & Goyal, 2000)
Agents form directed links to access information with decay. Tests whether
the resulting network follows the 80-20 "Law of the Few" pattern.

### Hub Prediction
Gradient Boosting (node-level) and Random Forest (network-level) classifiers
trained on simulation data to predict hub emergence from initial features.

## Results

Sample results from the analysis:

| Metric | Value |
|--------|-------|
| Hub prediction AUC | ~0.85 |
| Core node fraction (YouTube) | 3.3% |
| Top predictor | interest_mean |

See `results/figures/` for all visualisations and `results/tables/` for
detailed analysis outputs.

## Citation

If you use this code or data, please cite:

```
Sumanthiran, S. (2025). Interest Heterogeneity in Network Formation
on Large Online Platforms. [Capstone Project].
```

## License

This project is for academic purposes. Please contact the author for
reuse permissions.
