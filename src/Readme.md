# Confluence Protocol: Code Repository

This repository contains code for replicating and extending the Confluence Protocol experiments described in "Mathematical Framework for Semantic Convergence Topology" (Cook, 2025).

## Installation
```bash
pip install numpy scipy scikit-learn matplotlib
# Optional: for real embeddings
pip install sentence-transformers
```

## Quick Start
```bash
# Basic usage demo
python examples/basic_usage.py

# Replicate Experiment 1
python examples/replicate_experiment1.py

# Generate visualizations
python visualization/plot_trajectory.py
python visualization/plot_basins.py
```

## Repository Structure
```
├── examples/
│   ├── basic_usage.py              # Core functionality demo
│   └── replicate_experiment1.py    # Full replication script
├── visualization/
│   ├── plot_trajectory.py          # Phase convergence plots
│   └── plot_basins.py              # Basin structure visualization
├── data/
│   └── experiment1_raw.json        # Original experimental data
└── README.md
```

## Citation
```bibtex
@article{cook2025semantic,
  title={Mathematical Framework for Semantic Convergence Topology},
  author={Cook, Daniel Monroe},
  year={2025},
  note={U.S. Patent Application 63/912,870}
}
```

## License

Code: MIT  
Data: CC BY 4.0