
import torch
from transformers import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

#lr = 5e-5
def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    for batch in tqdm(train_loader):
        inputs, _, labels = batch
        inputs, _, labels = inputs.to(device), _.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, _)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate training accuracy
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)
    
    loss_incurred = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    '''

    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        

        
        training_losses.append(average_loss)

        # Calculate training accuracy for the epoch
        accuracy = correct_predictions / total_predictions
        training_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}: Loss {average_loss}, Accuracy {accuracy}")'''
    

    return loss_incurred, accuracy

def train2(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    # model.to(device)
    # training_losses = []
    # training_accuracies = []
    for batch in tqdm(train_loader):
        inputs, _, labels = batch
        inputs, _, labels = inputs.to(device), _.to(device), labels.to(device)

        optimizer.zero_grad()
        # inputs = inputs.float()
        # print("INPUT SHAPE: ",inputs.shape)
        outputs = model(inputs, _)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate training accuracy
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)
    
    loss_incurred = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    '''

    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        

        
        training_losses.append(average_loss)

        # Calculate training accuracy for the epoch
        accuracy = correct_predictions / total_predictions
        training_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}: Loss {average_loss}, Accuracy {accuracy}")'''
    

    return loss_incurred, accuracy

def train3(teacher_model, student_model, train_loader, optimizer, loss_fn, T,device, soft_target_loss_weight=0.25, ce_loss_weight=0.75):
    teacher_model.eval()
    student_model.train()
    # total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    # model.to(device)
    # training_losses = []
    # training_accuracies = []
    running_loss = 0.0

    for batch in tqdm(train_loader):
        inputs, _, labels = batch
        inputs, _, labels = inputs.to(device), _.to(device), labels.to(device)

        optimizer.zero_grad()
        # inputs = inputs.float()
        # print("INPUT SHAPE: ",inputs.shape)
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
        
        student_logits = student_model(inputs)

        soft_targets = torch.nn.functional.softmax(teacher_logits/T, dim=-1)
        soft_prob = torch.nn.functional.log_softmax(student_logits/T, dim=-1)

        soft_targets_loss = torch.sum(soft_targets*(soft_targets.log() - soft_prob)) / soft_prob.size()[0]*(T**2)

        label_loss = loss_fn(student_logits, labels)

        loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(student_logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)
        # outputs = model(inputs, _)
        # loss = loss_fn(outputs, labels)
        # loss.backward()
        # optimizer.step()

        # total_loss += loss.item()

        # # Calculate training accuracy
        # predictions = torch.argmax(outputs, dim=1)
        # correct_predictions += (predictions == labels).sum().item()
        # total_predictions += labels.size(0)
    
    loss_incurred = running_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    # return loss_incurred, accuracy
    
    '''

    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        

        
        training_losses.append(average_loss)

        # Calculate training accuracy for the epoch
        accuracy = correct_predictions / total_predictions
        training_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}: Loss {average_loss}, Accuracy {accuracy}")'''
    

    return loss_incurred, accuracy


def evaluate(model, val_loader,criterion, device):
    model.eval()
    model.to(device)
    # outputs_list = []

    validation_correct = 0
    validation_total = 0
    total_loss = 0


    with torch.no_grad():
        for batch in val_loader:
            inputs, _, labels = batch
            inputs,_, labels = inputs.to(device),_.to(device), labels.to(device)
            outputs = model(inputs, _)
            loss = criterion(outputs, labels)
            total_loss += loss.item()


            # outputs_list.append(outputs)
            predictions = torch.argmax(outputs, dim=1)
            validation_correct += (predictions == labels).sum().item()
            validation_total += labels.size(0)

    accuracy = validation_correct / validation_total
    # print(f"Validation Accuracy: {accuracy}")
    return total_loss/len(val_loader), accuracy



import os
# import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_metrics(training_losses, validation_losses, training_accuracies, validation_accuracies):
    """
    Plots training and validation metrics and saves them as images in a 'plots' folder.

    Args:
    - training_losses (list): List of training losses.
    - validation_losses (list): List of validation losses.
    - training_accuracies (list): List of training accuracies.
    - validation_accuracies (list): List of validation accuracies.
    """

    # Create 'plots' directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Plotting training and validation losses
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/losses.png')  # Save as plots/losses.png
    plt.close()

    # Plotting training and validation accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('plots/accuracy.png')  # Save as plots/accuracy.png
    plt.close()

