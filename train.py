
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.label_encoder = LabelEncoder()

    def encode_labels(self, labels):
        '''
        Encode labels using LabelEncoder
        '''
        targets_encoded = self.label_encoder.fit_transform(labels)
        targets_encoded = torch.tensor(targets_encoded, dtype=torch.long).to(self.device)
        return targets_encoded

    def calculate_accuracy(self, outputs, targets):
        '''
        Calculate the accuracy given model outputs and target labels
        '''
        _, predicted = torch.max(outputs, dim=1)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)
        return accuracy

    def train_fn(self, data_loader, loss_fn):
        '''
        Training step
        '''
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0

        for images, labels in tqdm(data_loader):
            images = images.to(self.device)
            targets_encoded = self.encode_labels(labels)

            self.optimizer.zero_grad()
            output = self.model(images)
            loss = loss_fn(output, targets_encoded)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_accuracy += self.calculate_accuracy(output, targets_encoded)

        avg_loss = total_loss / len(data_loader)
        avg_accuracy = total_accuracy / len(data_loader)

        return avg_loss, avg_accuracy

    def eval_fn(self, data_loader, loss_fn):
        '''
        Evaluation step
        '''
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0

        with torch.no_grad():
            for images, labels in tqdm(data_loader):
                images = images.to(self.device)
                targets_encoded = self.encode_labels(labels)

                output = self.model(images)
                loss = loss_fn(output, targets_encoded)

                total_loss += loss.item()
                total_accuracy += self.calculate_accuracy(output, targets_encoded)

        avg_loss = total_loss / len(data_loader)
        avg_accuracy = total_accuracy / len(data_loader)

        return avg_loss, avg_accuracy

    def train_and_evaluate(self, epoch, train_loader, valid_loader, loss_fn):
        best_valid_loss = float('inf')
        best_valid_accuracy = 0.0

        train_losses = []
        train_accuracies = []
        valid_losses = []
        valid_accuracies = []

        for i in range(epoch):
            train_loss, train_accuracy = self.train_fn(train_loader, loss_fn)
            valid_loss, valid_accuracy = self.eval_fn(valid_loader, loss_fn)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

            if valid_loss < best_valid_loss:
                torch.save(self.model.state_dict(), 'best_model.pt')
                print("SAVED-MODEL")
                best_valid_loss = valid_loss
                best_valid_accuracy = valid_accuracy

            print(f"Epoch: {i+1} Train loss: {train_loss:.4f} Train accuracy: {train_accuracy:.4f} Valid loss: {valid_loss:.4f} Valid accuracy: {valid_accuracy:.4f}")

            # Update the learning rate based on validation loss
            self.scheduler.step(valid_loss)

        print(f"Best Validation Accuracy: {best_valid_accuracy:.4f}")

        # Plot the loss and accuracies
        epochs = range(1, epoch + 1)

        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_losses, 'g', label='Training Loss')
        plt.plot(epochs, valid_losses, 'b', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_accuracies, 'r', label='Training Accuracy')
        plt.plot(epochs, valid_accuracies, 'c', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
