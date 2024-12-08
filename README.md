# URV NEC Assignment 1

Mark Safronov

MESIIA 2024-2025

Dataset is [Restaurant Revenue Prediction Dataset](https://www.kaggle.com/datasets/anthonytherrien/restaurant-revenue-prediction-dataset) from Kaggle.

The task is formally explained in the [attached PDF handout](./A1.pdf).

The solution is four Jupyter notebooks and two Python files.

1. [01-prepare-data.ipynb](01-prepare-data.ipynb) - notebook which analyses and transforms the [source dataset file](./data/restaurant_data.csv)

2. [02-use-neural-network.ipynb](02-use-neural-network.ipynb) - analysis of the custom NeuralNet class

3. [03-compare-with.ipynb](03-compare-with.ipynb) - regression on the [reduced dataset](./data/reduced_dataset.csv) using the Scikit and Pytorch

4. [03-compare-with.fulldata.ipynb](03-compare-with.fulldata.ipynb) - same as previous, but on the [full dataset](./data/full_dataset.csv)

5. [NeuralNet class](NeuralNet.py) - custom homemade neural network implementation using fully online backgropagation without cross-validation

6. [activation_functions.py](activation_functions.py) - helper module with the activation functions usable for the NeuralNet class.

