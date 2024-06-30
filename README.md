# The Pitfalls of Chasing Accuracy by Shuffling Dataset Splits

## Introduction
This repository explores the concept of repeatedly shuffling dataset splits to achieve the highest accuracy. It highlights the potential benefits and significant drawbacks of this approach.

## Structure
- `pitfalls_of_shuffling.ipynb`: A Jupyter Notebook with detailed explanations and code implementation.
- `src/`: Contains the main script and utility functions.
- `data/`: Includes a sample dataset for demonstration purposes.
- `results/`: Logs the metrics of the best split found.

## Running the Code
1. Clone the repository.
2. Install the required dependencies: `pip install -r requirements.txt`.
3. Run the main script: `python src/main.py`.

## Conclusion
While shuffling dataset splits to find the highest accuracy might seem beneficial, it is essential to understand the potential pitfalls and use robust validation techniques for reliable model performance evaluation.
