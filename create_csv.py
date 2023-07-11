import os
import pandas as pd


def create_dataframe(image_path):

    ''' This function creates a dataframe to store images path for each folder and the Label for each image based on the folder name'''

    # Create a list to store the path of each image
    image_list = []

    # Create a list to store the label of each image
    label_list = []

    # Loop through each folder in the image path
    for folder in os.listdir(image_path):
            
            # Loop through each image in the folder
            for image in os.listdir(os.path.join(image_path, folder)):
    
                # Append the image path to the image list
                image_list.append(os.path.join(image_path, folder, image))
    
                # Append the label to the label list
                label_list.append(folder)

    # Create a dataframe to store the image path and label
    df = pd.DataFrame({'image_path': image_list, 'label': label_list})
    
    # Save the dataframe as a csv file
    df.to_csv('dataframe.csv', index=False)