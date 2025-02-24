import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBM


class BaseClassifier(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_name(self):
        pass


class SimpleCNNClassifier(BaseClassifier):
    def __init__(self, input_shape):
        super().__init__()
        self.name = "simple_cnn"
        
        h, w = input_shape
        n_pools = min(
            3,
            min(
                int(math.log2(h/2)),
                int(math.log2(w/2))
            )
        )
        
        layers = []
        in_channels = 1
        channels = [16, 32, 64][:n_pools + 1]
        
        for i in range(n_pools):
            layers.extend([
                nn.Conv2d(in_channels, channels[i], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i]),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_channels = channels[i]
        
        layers.extend([
            nn.Conv2d(in_channels, channels[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU()
        ])
        
        self.conv_layers = nn.Sequential(*layers)
        
        with torch.no_grad():
            x = torch.zeros(1, 1, input_shape[0], input_shape[1])
            x = self.conv_layers(x)
            flat_size = x.view(1, -1).shape[1]
        
        hidden_size = min(256, flat_size)
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_name(self):
        return self.name


class RandomForestClassifier(BaseClassifier):
    def __init__(self, input_shape, n_estimators=100, max_depth=None, random_state=42):
        super().__init__()
        self.name = "random_forest"
        self.input_shape = input_shape
        
        # Initialize the sklearn RandomForest model
        self.rf_model = SklearnRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
            class_weight='balanced'
        )
        
        # Flag to track if the model has been trained
        self.is_trained = False
        
        # For compatibility with PyTorch's state_dict
        self.dummy_param = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        Forward pass for the RandomForest classifier.
        
        Args:
            x: Input tensor of shape [batch_size, height, width]
            
        Returns:
            Tensor of shape [batch_size, 2] with class logits
        """
        if not self.is_trained:
            # Return random predictions before training
            batch_size = x.shape[0]
            return torch.rand(batch_size, 2, device=x.device)
        
        # Move to CPU for sklearn
        if x.device.type != 'cpu':
            x = x.cpu()
        
        # Flatten the 2D input to 1D
        x_flat = x.view(x.size(0), -1).numpy()
        
        # Get predictions from the random forest
        proba = self.rf_model.predict_proba(x_flat)
        
        # Convert back to torch tensor
        logits = torch.from_numpy(proba).float()
        
        # Move back to the original device
        logits = logits.to(x.device)
        
        return logits
    
    def fit(self, X, y):
        """
        Train the random forest model.
        
        Args:
            X: Input features tensor [n_samples, height, width]
            y: Target labels tensor [n_samples]
        """
        # Flatten the 2D input to 1D
        X_flat = X.view(X.size(0), -1)
        
        # Move to CPU for sklearn
        if X.device.type != 'cpu':
            X_flat = X_flat.cpu()
            y = y.cpu()
        
        # Convert to numpy arrays
        X_np = X_flat.numpy()
        y_np = y.numpy()
        
        # Train the model
        self.rf_model.fit(X_np, y_np)
        self.is_trained = True
        
        return self
    
    def load_state_dict(self, state_dict):
        """
        Load the model state from a dictionary.
        For RandomForest, we need to handle the sklearn model separately.
        """
        import pickle
        if 'rf_model' in state_dict:
            self.rf_model = pickle.loads(state_dict['rf_model'])
            self.is_trained = True
        
        # Load any PyTorch parameters (dummy in this case)
        super_dict = {k: v for k, v in state_dict.items() if k != 'rf_model'}
        super().load_state_dict(super_dict, strict=False)
    
    def state_dict(self):
        """
        Return the model state as a dictionary.
        For RandomForest, we need to handle the sklearn model separately.
        """
        import pickle
        state = super().state_dict()
        state['rf_model'] = pickle.dumps(self.rf_model)
        return state
    
    def get_name(self):
        return self.name


class GradientBoostingClassifier(BaseClassifier):
    def __init__(self, input_shape, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        super().__init__()
        self.name = "gradient_boosting"
        self.input_shape = input_shape
        
        # Initialize the sklearn GradientBoosting model
        self.gbm_model = SklearnGBM(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            verbose=1
        )
        
        # Flag to track if the model has been trained
        self.is_trained = False
        
        # For compatibility with PyTorch's state_dict
        self.dummy_param = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        Forward pass for the GradientBoosting classifier.
        
        Args:
            x: Input tensor of shape [batch_size, height, width]
            
        Returns:
            Tensor of shape [batch_size, 2] with class logits
        """
        if not self.is_trained:
            # Return random predictions before training
            batch_size = x.shape[0]
            return torch.rand(batch_size, 2, device=x.device)
        
        # Move to CPU for sklearn
        if x.device.type != 'cpu':
            x = x.cpu()
        
        # Flatten the 2D input to 1D
        x_flat = x.view(x.size(0), -1).numpy()
        
        # Get predictions from the GBM
        proba = self.gbm_model.predict_proba(x_flat)
        
        # Convert back to torch tensor
        logits = torch.from_numpy(proba).float()
        
        # Move back to the original device
        logits = logits.to(x.device)
        
        return logits
    
    def fit(self, X, y):
        """
        Train the gradient boosting model.
        
        Args:
            X: Input features tensor [n_samples, height, width]
            y: Target labels tensor [n_samples]
        """
        # Flatten the 2D input to 1D
        X_flat = X.view(X.size(0), -1)
        
        # Move to CPU for sklearn
        if X.device.type != 'cpu':
            X_flat = X_flat.cpu()
            y = y.cpu()
        
        # Convert to numpy arrays
        X_np = X_flat.numpy()
        y_np = y.numpy()
        
        # Train the model
        self.gbm_model.fit(X_np, y_np)
        self.is_trained = True
        
        return self
    
    def load_state_dict(self, state_dict):
        """
        Load the model state from a dictionary.
        For GBM, we need to handle the sklearn model separately.
        """
        import pickle
        if 'gbm_model' in state_dict:
            self.gbm_model = pickle.loads(state_dict['gbm_model'])
            self.is_trained = True
        
        # Load any PyTorch parameters (dummy in this case)
        super_dict = {k: v for k, v in state_dict.items() if k != 'gbm_model'}
        super().load_state_dict(super_dict, strict=False)
    
    def state_dict(self):
        """
        Return the model state as a dictionary.
        For GBM, we need to handle the sklearn model separately.
        """
        import pickle
        state = super().state_dict()
        state['gbm_model'] = pickle.dumps(self.gbm_model)
        return state
    
    def get_name(self):
        return self.name