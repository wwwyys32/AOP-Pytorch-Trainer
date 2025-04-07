# AOP Annotations - A simple implementation for Pytorch Trainer

This document provides a detailed explanation of the PyTorch training and evaluation decorators implemented in the provided code. These decorators are designed to simplify the process of training, validating, and testing PyTorch models by encapsulating common functionality and allowing for reusable and modular code.

## Table of Contents
1. [Introduction](#introduction)
2. [Decorators](#decorators)
    - [Train Decorator](#train-decorator)
    - [Validation Decorator](#validation-decorator)
    - [Test Decorator](#test-decorator)
3. [Usage Example](#usage-example)
4. [Implementation Details](#implementation-details)
5. [Conclusion](#conclusion)

## Introduction

Training, validating, and testing machine learning models are fundamental steps in the development process. These steps often involve repetitive code patterns, such as iterating over data loaders, computing losses, and updating model parameters. To streamline this process, we have implemented three decorators: `Train`, `Validation`, and `Test`. These decorators encapsulate the common logic for each step, allowing users to focus on defining their models and custom logic while leveraging reusable code.

## Decorators

### Train Decorator

The `Train` decorator is designed to handle the training process of a PyTorch model. It takes the following parameters:
- `optimizer`: The optimizer instance used to update the model's parameters.
- `criterion`: The loss function used to compute the training loss.
- `epochs`: The number of training epochs.
- `validate_every`: The interval (in epochs) at which the validation function is called. Default is 1.
- `device`: The device (e.g., "cpu" or "cuda") on which the training is performed. Default is "cpu".

The decorator performs the following steps:
1. Moves the model to the specified device and sets it to training mode.
2. Iterates over the training data loader, computing the loss and updating the model parameters using the optimizer.
3. Prints the training loss after each epoch.
4. Calls the validation function at the specified interval.

### Validation Decorator

The `Validation` decorator is used to evaluate the model's performance on a validation dataset. It takes the following parameters:
- `criterion`: The loss function used to compute the validation loss.
- `device`: The device on which the validation is performed. Default is "cpu".

The decorator performs the following steps:
1. Moves the model to the specified device and sets it to evaluation mode.
2. Iterates over the validation data loader, computing the validation loss without updating the model parameters.
3. Prints the validation loss.

### Test Decorator

The `Test` decorator is used to evaluate the model's performance on a test dataset. It takes the following parameter:
- `device`: The device on which the testing is performed. Default is "cpu".

The decorator performs the following steps:
1. Moves the model to the specified device and sets it to evaluation mode.
2. Iterates over the test data loader, computing the test loss without updating the model parameters.
3. Prints the test loss.

## Usage Example

The following example demonstrates how to use the decorators to train, validate, and test a simple PyTorch model.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

x_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,))
x_val = torch.randn(20, 10)
y_val = torch.randint(0, 2, (20,))
x_test = torch.randn(30, 10)
y_test = torch.randint(0, 2, (30,))

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=10, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=10, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=10, shuffle=False)

model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(SimpleModel().parameters(), lr=0.01)


@Train(optimizer=optimizer, criterion=criterion, epochs=10, validate_every=1, device="cpu")
@Test(device="cpu")
def run_training(model, **kwargs):
    print("Training, validation, and testing completed!")

run_training(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, validate_func=Validation(criterion=criterion, device="cpu")(lambda *args, **kwargs: None))
```

In this example:
- The `Train` decorator handles the training process for 10 epochs.
- The `Validation` decorator evaluates the model on the validation dataset after `validate_every` epoch.
- The `Test` decorator evaluates the model on the test dataset after training is complete.

## Implementation Details

The decorators are implemented using Python's higher-order functions and closures. Each decorator takes a function as input and returns a new function that wraps the original function with additional functionality. The decorators are designed to be reusable and flexible, allowing users to easily integrate them into their PyTorch workflows. Additionally, users can freely modify the implementation details according to their specific needs.

## Conclusion

The `Train`, `Validation`, and `Test` decorators provide a convenient and modular way to handle the training, validation, and testing of PyTorch models. By encapsulating common logic, these decorators reduce code duplication and improve code readability. Users can focus on defining their models and custom logic while leveraging the reusable functionality provided by the decorators.
