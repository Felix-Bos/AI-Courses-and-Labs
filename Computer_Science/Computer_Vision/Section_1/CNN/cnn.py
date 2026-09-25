import torch

print("import terminés")


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(5, 5), stride=1, padding=2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(5, 5), stride=1, padding=2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(5, 5), stride=1, padding=2)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.fc4 = torch.nn.Linear(64 * 4 * 4, 1000)
        self.relu = torch.nn.ReLU()
        self.fc5 = torch.nn.Linear(1000, 10)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x): # dim x = (B, 3, 32, 32)
        x = self.pool1(self.relu(self.conv1(x))) # output dim = (B, 32, 16, 16) bc pool1
        x = self.pool2(self.relu(self.conv2(x))) # output dim = (B, 64, 8, 8)
        x = self.pool3(self.relu(self.conv3(x))) # output dim = (B, 64, 4, 4)
        x = x.view(-1, 64 * 4 * 4) # output dim = (B, 64*4*4)
        x = self.relu(self.fc4(x)) # output dim = (B, 1000)
        output = self.softmax(self.fc5(x)) # output dim = (B, 10)
        return output
    
    
        
        

        
        
        