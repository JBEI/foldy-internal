import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm

class MLPRegressorWithRankedLoss(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', 
                 max_iter=200, learning_rate=0.001, batch_size='auto', 
                 margin=1.0, early_stopping=False, random_state=None,device=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.margin = margin
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.early_stopping = early_stopping
        # Determine device (GPU or CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def _get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Activation {self.activation} not supported")
            
    def _build_model(self, input_dim):
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _pairwise_ranking_loss(self, outputs, targets):
        # Create all possible pairs
        n = outputs.size(0)
        pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                if targets[i] > targets[j]:
                    pairs.append((i, j))
                elif targets[j] > targets[i]:
                    pairs.append((j, i))
        
        if not pairs:
            return torch.tensor(0.0, requires_grad=True)
        
        loss = 0
        for higher, lower in pairs:
            # RankNet loss: max(0, margin - (score_higher - score_lower))
            diff = outputs[higher] - outputs[lower]
            loss += torch.max(torch.tensor(0.0), self.margin - diff)
            
        return loss / len(pairs)
    
    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
  
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Move tensors to the appropriate device
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        
        self.model_ = self._build_model(X.shape[1]).to(self.device)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        # Rest of the fit method remains the same, but ensure tensors are on the device
        n_samples = X.shape[0]
        if self.batch_size =='auto':
            self.batch_size = min(200,n_samples)
        indices = np.arange(n_samples)
        current_loss
        for epoch in tqdm(range(self.max_iter), desc=f"Epochs on {self.device}", leave=False):
            np.random.shuffle(indices)
            total_loss = 0

            for start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                X_batch = X_tensor[batch_indices]
                y_batch = y_tensor[batch_indices]
                
                # Forward pass
                outputs = self.model_(X_batch)
                
                # Compute loss
                loss = self._pairwise_ranking_loss(outputs, y_batch)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model_.eval()
        with torch.no_grad():
            predictions = self.model_(X_tensor)
        
        # Move predictions back to CPU for numpy conversion
        return predictions.cpu().numpy().flatten()