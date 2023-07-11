
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix



class ImageClassifier:
    def __init__(self):
        self.lb = LabelEncoder()
        self.true_labels = []
        self.predicted_labels = []
        self.predicted_classes = []  # Added attribute to store predicted classes

    def predict(self, test_loader, efficientnet_model, device):
        for image, label in test_loader:
            image = image.to(device)
            
            # Perform forward pass and obtain predicted batch
            predicted_batch = efficientnet_model(image).detach().cpu().numpy()
            
            # Encode true labels
            label = self.lb.fit_transform(label)
            
            # Convert true labels to tensors
            true_batch = torch.tensor(label, dtype=torch.long).to(device).cpu()

            # Extend predicted and true labels
            self.predicted_labels.extend(predicted_batch)
            self.true_labels.extend(true_batch)

        # Convert labels to numpy arrays
        self.true_labels = np.array(self.true_labels)
        self.predicted_labels = np.array(self.predicted_labels)
        
        # Get predicted classes by finding the index of the maximum value
        self.predicted_classes = np.argmax(self.predicted_labels, axis=1)  # Assuming it's a multiclass problem

        # Convert encoded labels back to original labels
        self.true_labels = self.lb.inverse_transform(self.true_labels)
        self.predicted_classes = self.lb.inverse_transform(self.predicted_classes)

        return self.true_labels, self.predicted_classes

    def print_classification_report(self, true_labels, predicted_classes):
        # Generate classification report
        report = classification_report(true_labels, predicted_classes)
        
        # Print classification report
        print(report)

    def plot_confusion_matrix(self, true_labels, predicted_classes):
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_classes)

        # Get class labels
        class_labels = self.lb.classes_

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

    def display_predicted_images(self, test_loader, num_images=5):
        # Select a few images from the test loader to display
        selected_indices = np.random.choice(len(test_loader.dataset), num_images, replace=False)

        # Create a subplot grid for displaying the images
        fig, axes = plt.subplots(1, num_images, figsize=(12, 4))

        # Iterate over the selected indices and display images with predicted and true labels
        for i, idx in enumerate(selected_indices):
            image, label = test_loader.dataset[idx]
            image = np.transpose(image, (0, 1, 2))  # Rearrange dimensions to (height, width, channels)
            predicted_label = self.predicted_classes[idx]  # Access predicted classes from the attribute

            # Display the image
            axes[i].imshow(image)
            axes[i].axis('off')

            # Annotate the predicted and true labels
            axes[i].set_title(f"Predicted: {predicted_label}\nTrue: {label}", fontsize=10)

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.show()
