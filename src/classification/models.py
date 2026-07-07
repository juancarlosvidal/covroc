'''import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, input_features):
        """Multilayer Perceptron for classification"""
        super(RegressionModel, self).__init__()
        self.layers = nn.Sequential(
            # nn.Linear(10, 64),
            nn.Linear(input_features, 6),
            nn.ReLU(),
            # nn.BatchNorm1d(8),
            # nn.Dropout(p=0.20),
            #nn.Linear(8, 4),
            #nn.ReLU(),
            # nn.BatchNorm1d(16),
            # nn.Dropout(p=0.20),
            # nn.Linear(16, 8),
            # nn.ReLU(),
            # nn.BatchNorm1d(8),
            # nn.Dropout(p=0.20),
            # nn.Linear(16, 8),
            # nn.ReLU(),
            # nn.BatchNorm1d(8),
            # nn.Dropout(p=0.20),
            # nn.Linear(8, 4),
            # nn.ReLU(),
            # nn.BatchNorm1d(4),
            # nn.Dropout(p=0.20),
            nn.Linear(6, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.3, std= 1)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)

'''
import torch
import torch.nn as nn

'''class RegressionModel(nn.Module):
    def __init__(self, input_features):
        """Enhanced Multilayer Perceptron for regression with an explicit intercept"""
        super(RegressionModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_features, 32),  # First layer
            nn.ReLU(),
            nn.BatchNorm1d(32),  # Batch normalization
            nn.Dropout(p=0.2),   # Dropout to prevent overfitting
            
            nn.Linear(32, 16),   # Second layer
            nn.ReLU(),
            nn.BatchNorm1d(16),  # Batch normalization
            nn.Dropout(p=0.2),   # Dropout
            
            nn.Linear(16, 8),    # Third layer
            nn.ReLU(),
            nn.BatchNorm1d(8),   # Batch normalization
            
            nn.Linear(8, 1)      # Output layer for regression
        )
        
        # Apply custom weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using normal distribution and biases to zero"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)  # Normal distribution
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)  # Biases set to zero

    def forward(self, x):
        """Forward pass through the network"""
        output = self.layers(x)
        # Add intercept (if needed), but generally handled by the last Linear layer's bias
        intercept = self.layers[-1].bias.item()  # Getting the intercept value from the last layer
        return output + intercept  # Adding intercept to the output, if desired
'''

import torch
import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self, input_features):
        super(RegressionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        output = self.layers(x)
        return output

