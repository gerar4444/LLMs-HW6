from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_predictions(test_labels, test_predictions, ordered_labels):
    """
    Evaluate model predictions and produce metrics and visualizations.

    Args:
        test_labels (list): Ground truth labels.
        test_predictions (list): Predicted labels.
        ordered_labels (list): List of label names in order.

    Returns:
        dict: Evaluation metrics (accuracy, MAE, discounted gain, and classification report).
    """
    # Step 1: Accuracy
    accuracy = accuracy_score(test_labels, test_predictions)
    
    # Step 2: Classification Report
    class_report = classification_report(test_labels, test_predictions, target_names=ordered_labels)
    
    # Step 3: Mean Absolute Error (MAE)
    mae = mean_absolute_error(test_labels, test_predictions)
    
    # Step 4: Discounted Gain
    discounted_gain = sum(1 / (1 + abs(pred - true)) for pred, true in zip(test_predictions, test_labels)) / len(test_labels)
    
    # Step 5: Confusion Matrix
    conf_matrix = confusion_matrix(test_labels, test_predictions)
    
    # Print Metrics
    print("\nClassification Report:")
    print(class_report)
    
    print("\nAccuracy:", accuracy)
    print("Mean Absolute Error (MAE):", mae)
    print("Discounted Gain:", discounted_gain)
    
    # Display First 20 Predicted vs Real Labels
    print("\nPredicted vs Real Labels (First 20 Examples):")
    for i, (pred, true) in enumerate(zip(test_predictions[:20], test_labels[:20])):
        print(f"Example {i + 1}: Predicted: {pred} | Real: {true}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=ordered_labels, yticklabels=ordered_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
    
    # Return Metrics
    return {
        "accuracy": accuracy,
        "mae": mae,
        "discounted_gain": discounted_gain,
        "classification_report": class_report,
    }
