

import cv2
import numpy as np 
import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch 
from torchvision import datasets
from torch.utils.data import Dataset,WeightedRandomSampler

from collections import Counter
from sklearn.preprocessing import LabelEncoder



class ChestXRay(Dataset):
    
    """
    Chest X-Ray dataset.
    Args:
    df (string): : DataFrame containing image path and labels.
    augmentations (callable, optional):Augmentation pipeline to be applied to the data.
                Defaults to None.
    """
    
    
    # Initialize the  class.
    def __init__(self, df, augmentations=None):
        
        self.df = df
        self.augmentations = augmentations
    
        
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            image (torch.Tensor): Preprocessed image tensor.
            label (torch.Tensor): Preprocessed mask tensor.
        """
        row = self.df.iloc[idx]
        
        image_path = row['image_path']
        
        # Read the image file and convert color space to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # get the label
        label = row['label']
        
        if self.augmentations:
            
            # Apply augmentations to the image
            data = self.augmentations(image=image)
            image = data['image']
           
        image = torch.Tensor(image) / 255.0
        
        
        
        return image, label
        
        
def get_train_augs():
    """
    Define the augmentation pipeline for training data.
    """
    return A.Compose([
        A.Resize(224,224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=1),
  
        
    ])  

def get_valid_augs():
    """
    Define the augmentation pipeline for validation data.
    """
    return A.Compose([
        A.Resize(224,224),
  
    ])






def create_weighted_sampler(dataset):
    """
    Dealing with imbalanced dataset to improve the model's performance on the under-represented class.
    """
    # Extract the target labels from the dataset
    targets = [label for _, label in tqdm.tqdm(dataset)]

    # Encode the labels into numerical values
    label_encoder = LabelEncoder()
    targets_encoded = label_encoder.fit_transform(targets)

    # Count the occurrences of each unique label in the target labels
    class_counts = Counter(targets_encoded)

    # Calculate the class weights by taking the reciprocal of the class counts
    class_weights = 1.0 / np.array([class_counts[label] for label in range(len(class_counts))])

    # Assign weights to each sample based on its label
    weights = [class_weights[label] for label in targets_encoded]

    # Create a weighted random sampler using the computed weights
    sampler = WeightedRandomSampler(weights, len(weights))

    return sampler
