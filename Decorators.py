import torch
from typing import Callable

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
                print("Training -", end=" ")
                running_loss = 0.0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

                if (epoch + 1) % validate_every == 0:
                    validate_func(*args, **kwargs)

            return func(*args, **kwargs)
        return wrapper
    return decorator_train

def Validation(criterion: Callable, device: str = "cpu")->Callable:
    def decorator_validate(func: Callable):
        def wrapper(*args, **kwargs):
            model = args[0]  # 假设模型是第一个参数
            val_loader = kwargs.get("val_loader")
            if val_loader is None:
                raise ValueError("val_loader must be provided as a keyword argument")

            model.to(device)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
            return func(*args, **kwargs)
        return wrapper
    return decorator_validate

def Test(device: str = "cpu")->Callable:
    def decorator_test(func: Callable):
        def wrapper(*args, **kwargs):
            model = args[0]  # 假设模型是第一个参数
            test_loader = kwargs.get("test_loader")
            if test_loader is None:
                raise ValueError("test_loader must be provided as a keyword argument")

            model.to(device)
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()

            print(f"Test Loss: {test_loss / len(test_loader):.4f}")
            return func(*args, **kwargs)
        return wrapper
    return decorator_test


if __name__ == '__main__':
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