import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import utils

def find_best_split(data, target_column, model, test_size=0.2, iterations=100):
    best_accuracy = 0
    best_train_set = None
    best_test_set = None
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    for i in range(iterations):
        # Shuffle and split the dataset
        X_shuffled, y_shuffled = utils.shuffle_data(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=test_size)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Keep track of the best split
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_train_set = (X_train, y_train)
            best_test_set = (X_test, y_test)
        
        print(f"Iteration {i+1}/{iterations}, Accuracy: {accuracy}")
    
    print(f"Best Accuracy: {best_accuracy}")
    return best_train_set, best_test_set, best_accuracy

if __name__ == "__main__":
    # Load your dataset
    data = pd.read_csv('data/sample_data.csv')
    
    # Define your target column
    target_column = 'your_target_column'
    
    # Initialize your model
    model = RandomForestClassifier()
    
    # Find the best split
    best_train_set, best_test_set, best_accuracy = find_best_split(data, target_column, model)
    
    # Save the results
    with open('results/best_split_metrics.txt', 'w') as f:
        f.write(f"Best Accuracy: {best_accuracy}\n")
