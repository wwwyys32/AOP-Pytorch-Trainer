import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Decorators import Train, Validation, Test

if __name__ == '__main__':
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 100)
            self.fc1 = nn.Linear(100, 2)

        def forward(self, x):
            return self.fc1(self.fc(x))

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
    @Test(criterion=criterion, device="cpu")
    def run_training(model, **kwargs):
        print("Training, validation, and testing completed!")

    run_training(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, validate_func=Validation(criterion=criterion, device="cpu")(lambda model, **kwargs: None))