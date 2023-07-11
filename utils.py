
import cv2
import random
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

import csv
import os
from sklearn.model_selection import train_test_split



def show_multiple_images(dataset, num_images):
    # Create a figure and axes for subplots
    fig, axes = plt.subplots(1, num_images, figsize=(15, 15))

    # Iterate over the number of images
    for i in range(num_images):
        # Choose a random index
        random_idx = random.randint(0, len(dataset) - 1)

        # Retrieve the image and label using the random index
        image, label = dataset[random_idx]

        # Display the image and label in the corresponding subplot
        axes[i].imshow(image)
        axes[i].set_title('Label: ' + label)
        axes[i].axis('off')

    # Adjust the layout and spacing
    plt.tight_layout()

    # Show the figure with multiple images
    plt.show()


def show_labels_distribution(dataframe):

    # Assuming you have a dataset named 'df' with a label column 'label'
    label_counts = dataframe['label'].value_counts()

    # Customizing the pie chart
    plt.pie(label_counts, labels=label_counts.index,autopct='%1.1f%%',shadow=True)

    # Adding a title and displaying the plot
    plt.title('Distribution of Labels')
    plt.show()
