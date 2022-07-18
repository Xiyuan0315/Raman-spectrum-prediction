import torch
import torch.nn as nn

# MLP
class MLP_CE(nn.Module):
    def __init__(self):
        super(MLP_CE, self).__init__()
        layers = []
        n_nodes_1=32
        layers.append(nn.Linear(in_features=779, out_features=n_nodes_1, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=n_nodes_1, out_features=n_nodes_1, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=n_nodes_1, out_features=n_nodes_1, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=n_nodes_1, out_features=2, bias=True))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        y = self.mlp(x)
        return y

class MLP_CE_attention(nn.Module):
    def __init__(self):
        super(MLP_CE_attention, self).__init__()
        layers = []
        n_nodes_1=32
        layers.append(nn.Linear(in_features=779, out_features=n_nodes_1, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=n_nodes_1, out_features=n_nodes_1, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=n_nodes_1, out_features=n_nodes_1, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=n_nodes_1, out_features=2, bias=True))
        self.mlp = nn.Sequential(*layers)
        
        layers = []
        n_nodes_2=128
        layers.append(nn.Linear(in_features=779, out_features=n_nodes_2, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=n_nodes_2, out_features=n_nodes_2, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=n_nodes_2, out_features=779, bias=True))
        self.attention = nn.Sequential(*layers)
        
    def forward(self, x):
        y = self.mlp(x*self.attention(x))
        return y
    
class MLP_CE_2(nn.Module):
    def __init__(self):
        super(MLP_CE_2, self).__init__()
        layers = []
        n_nodes_1=32
        layers.append(nn.Linear(in_features=779, out_features=n_nodes_1, bias=True))
        layers.append(torch.nn.BatchNorm1d(n_nodes_1))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(in_features=n_nodes_1, out_features=n_nodes_1, bias=True))
        layers.append(torch.nn.BatchNorm1d(n_nodes_1))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(in_features=n_nodes_1, out_features=n_nodes_1, bias=True))
        layers.append(torch.nn.BatchNorm1d(n_nodes_1))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(in_features=n_nodes_1, out_features=2, bias=True))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        y = self.mlp(x)
        return y


# CNN
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class CNN_CE(nn.Module):
    def __init__(self):
        super(CNN_CE, self).__init__()
        n_c = 32
        self.feat_extractor = nn.Sequential(*[
                                nn.Conv1d(in_channels=1, out_channels=n_c//2, kernel_size=11, stride=5, padding=0, bias=False),
                                nn.ReLU(),
                                nn.Conv1d(in_channels=n_c//2, out_channels=n_c, kernel_size=7, stride=3, padding=0, bias=False),
                                nn.ReLU(),
                                ])
        self.feat_mapping = nn.Sequential(*[
                                nn.Conv1d(in_channels=n_c, out_channels=n_c, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.ReLU(),
                                nn.Conv1d(in_channels=n_c, out_channels=n_c, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.ReLU(),
                                ])
        self.classifier = nn.Sequential(*[
                            nn.AdaptiveAvgPool1d(1), 
                            Flatten(), 
                            nn.Linear(in_features=n_c, out_features=2, bias=True),
                            ])
    def forward(self, x):
        out = self.feat_extractor(x)
        out = out + self.feat_mapping(out)
        y = self.classifier(out)
        return y

class CNN_CE_attention(nn.Module):
    def __init__(self):
        super(CNN_CE_attention, self).__init__()
        n_c = 32
        self.feat_extractor = nn.Sequential(*[
                                nn.Conv1d(in_channels=1, out_channels=n_c//2, kernel_size=11, stride=5, padding=0, bias=False),
                                nn.ReLU(),
                                nn.Conv1d(in_channels=n_c//2, out_channels=n_c, kernel_size=7, stride=3, padding=0, bias=False),
                                nn.ReLU(),
                                ])
        self.feat_mapping = nn.Sequential(*[
                                nn.Conv1d(in_channels=n_c, out_channels=n_c, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.ReLU(),
                                nn.Conv1d(in_channels=n_c, out_channels=n_c, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.ReLU(),
                                ])
        self.classifier = nn.Sequential(*[
                            nn.AdaptiveAvgPool1d(1), 
                            Flatten(), 
                            nn.Linear(in_features=n_c, out_features=2, bias=True),
                            ])
        
        self.attention = nn.Sequential(*[
                                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.ReLU(),
                                nn.Conv1d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.ReLU(),])
    def forward(self, x):
        x = x * self.attention(x)
        out = self.feat_extractor(x)
        out = out + self.feat_mapping(out)
        y = self.classifier(out)
        return y
    
class CNN_CE_2(nn.Module):
    def __init__(self):
        super(CNN_CE_2, self).__init__()
        n_c = 64
        self.feat_extractor = nn.Sequential(*[
                                nn.Conv1d(in_channels=1, out_channels=n_c, kernel_size=11, stride=5, padding=0, bias=False),
                                nn.BatchNorm1d(n_c),
                                nn.ReLU(),
                                nn.Conv1d(in_channels=n_c, out_channels=n_c, kernel_size=7, stride=3, padding=0, bias=False),
                                nn.BatchNorm1d(n_c),
                                nn.ReLU(),
                                ])
        self.feat_mapping = nn.Sequential(*[
                                nn.Conv1d(in_channels=n_c, out_channels=n_c, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm1d(n_c),
                                nn.ReLU(),
                                nn.Conv1d(in_channels=n_c, out_channels=n_c, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm1d(n_c),
                                nn.ReLU(),
                                ])
        self.classifier = nn.Sequential(*[
                            nn.AdaptiveAvgPool1d(1), 
                            Flatten(), 
                            nn.Linear(in_features=n_c, out_features=2, bias=True),
                            ])
    def forward(self, x):
        out = self.feat_extractor(x)
        out = out + self.feat_mapping(out)
        y = self.classifier(out)
        return y
    
if __name__=='__main__':
    input = torch.rand(100,1,779)
    net = CNN()
    output = net(input)
    import pdb; pdb.set_trace()
    pass