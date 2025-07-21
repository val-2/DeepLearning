# PikaPikaGen

A deep learning project for Pokemon generation using attention mechanisms and CNN.

## Installation


### Step 1: Clone the Repository

```bash
git clone https://github.com/val-2/DeepLearning.git
cd DeepLearning
```

### Step 2: Install Dependencies

This project uses `pyproject.toml` for dependency management. You can install dependencies using `pip`.

```bash
pip install -e .
```

## Running the Application

### Gradio Demo

To run the interactive Pokemon generation demo:

```bash
python pikapikagen/gradio_demo.py
```

This will start a Gradio web interface where you can:
- Generate Pokemon images from text descriptions
- Visualize attention mechanisms
- Explore the model's capabilities interactively

The demo will be available at `http://localhost:7860` by default.

### Jupyter Notebooks

The project includes Jupyter notebooks for training and experimentation:

- `PikaPikaTraining.ipynb` - Used to clone the repository in Jupyter environments
- `pikapikagen/PikaPikaGen.ipynb` - Main training notebook

To run the notebooks:

```bash
jupyter lab
```
