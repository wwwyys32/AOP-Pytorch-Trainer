import datetime
import torch
import os
import pandas as pd
from typing import Callable

run_number = None

def LogLoss():
    def decorator_log(func: Callable):
        def wrapper(*args, **kwargs):
            global run_number

            STATE = args[0]
            loss_item = kwargs.get('loss')

            if loss_item is None:
                raise ValueError('loss_item must be provided as a keyword argument')

            output_dir = "./outputs"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            run_folders = [d for d in os.listdir(output_dir) if d.startswith("run")]
            if not run_number:
                run_number = len(run_folders) + 1
            run_folder = os.path.join(output_dir, f"run{run_number}")
            os.makedirs(run_folder, exist_ok=True)

            log_file = os.path.join(run_folder, "log.xlsx")

            if not os.path.exists(log_file):
                log_df = pd.DataFrame(columns=["State", "Loss", "Time"])
            else:
                log_df = pd.read_excel(log_file)

            log_df = log_df.dropna(axis=1, how='all')

            new_log = pd.DataFrame({"State": [STATE], "Loss": [loss_item], "Time": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")]})
            log_df = pd.concat([log_df, new_log], ignore_index=True)

            log_df.to_excel(log_file, index=False)

            return func(*args, **kwargs)
        return wrapper
    return decorator_log
def LogAccuracy():
    def decorator_log(func: Callable):
        def wrapper(*args, **kwargs):
            global run_number

            STATE = args[0]
            accuracy_item = kwargs.get('accuracy')

            if accuracy_item is None:
                raise ValueError('accuracy_item must be provided as a keyword argument')

            output_dir = "./outputs"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            run_folders = [d for d in os.listdir(output_dir) if d.startswith("run")]
            if not run_number:
                run_number = len(run_folders) + 1
            run_folder = os.path.join(output_dir, f"run{run_number}")
            os.makedirs(run_folder, exist_ok=True)

            log_file = os.path.join(run_folder, "log.xlsx")

            if not os.path.exists(log_file):
                log_df = pd.DataFrame(columns=["State", "Accuracy", "Time"])
            else:
                log_df = pd.read_excel(log_file)

            log_df = log_df.dropna(axis=1, how='all')

            new_log = pd.DataFrame({"State": [STATE], "Accuracy": [accuracy_item], "Time": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]]})
            log_df = pd.concat([log_df, new_log], ignore_index=True)

            log_df.to_excel(log_file, index=False)

            return func(*args, **kwargs)
        return wrapper
    return decorator_log

@LogAccuracy()
@LogLoss()
def Log(state: str, loss: float, accuracy: float):
    pass

def Train(optimizer: torch.optim.Optimizer, criterion: Callable, epochs: int, validate_every: int = 1, device: str = "cpu")->Callable:
    def decorator_train(func: Callable):
        def wrapper(*args, **kwargs):
            model = args[0]
            train_loader = kwargs.get("train_loader")
            if train_loader is None:
                raise ValueError("train_loader must be provided as a keyword argument")
            validate_func = kwargs.get("validate_func")
            if validate_func is None:
                raise ValueError("validate_func must be provided as a keyword argument")

            model.to(device)
            model.train()

            for epoch in range(epochs):
                running_loss = 0.0
                correct = 0
                total = 0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                train_accuracy = 100 * correct / total
                Log("Train", loss=running_loss / len(train_loader), accuracy=train_accuracy)

                if (epoch + 1) % validate_every == 0:
                    validate_func(*args, **kwargs)

            return func(*args, **kwargs)
        return wrapper
    return decorator_train

def Validation(criterion: Callable, device: str = "cpu")->Callable:
    def decorator_validate(func: Callable):
        def wrapper(*args, **kwargs):
            model = args[0]
            val_loader = kwargs.get("val_loader")
            if val_loader is None:
                raise ValueError("val_loader must be provided as a keyword argument")

            model.to(device)
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            train_accuracy = 100 * correct / total
            Log("Validation", loss=val_loss / len(val_loader), accuracy=train_accuracy)

            return func(*args, **kwargs)
        return wrapper
    return decorator_validate

def Test(criterion: Callable, device: str = "cpu")->Callable:
    def decorator_test(func: Callable):
        def wrapper(*args, **kwargs):
            model = args[0]
            test_loader = kwargs.get("test_loader")
            if test_loader is None:
                raise ValueError("test_loader must be provided as a keyword argument")

            model.to(device)
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            train_accuracy = 100 * correct / total
            Log("Test", loss=test_loss / len(test_loader), accuracy=train_accuracy)

            return func(*args, **kwargs)
        return wrapper
    return decorator_test