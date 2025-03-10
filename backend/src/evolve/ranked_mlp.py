import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

class MLPRegressorWithRankedLoss(BaseEstimator, RegressorMixin):
    
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', 
                 max_iter=200, solver='adam', 
                 learning_rate_init=0.001, power_t=0.5, learning_rate='constant',
                 batch_size='auto', margin=1.0, 
                 early_stopping=False, validation_fraction=0.1,
                 tol=1e-4, n_iter_no_change=10, random_state=None, device=None,
                 verbose=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.learning_rate_current = learning_rate_init
        self.power_t = power_t
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.margin = margin
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        self._estimator_type = "regressor"
        # Determine device (GPU or CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def _get_learning_rate(self,optimizer):
        #ID
        if self.learning_rate == 'constant':
            return self.learning_rate_init
        if not optimizer:
            if self.learning_rate == 'invscaling':
                self.scheduler = StepLR(optimizer, step_size=100,gamma=0.1)
            if self.learning_rate == 'adaptive':
                self.scheduler = ReduceLROnPlateau(optimizer,)
        else:
            return self.learning_rate_init

    def _get_optimizer(self,parameters):
        #ID w/ parameter edits
        if self.solver == 'adam':
            return optim.Adam(parameters, lr=self.learning_rate_current)
        if self.solver == 'sgd':
            return optim.SGD(parameters, lr=self.learning_rate_current)
        if self.solver == 'lbfgs':
            return optim.LBFGS(parameters, lr=self.learning_rate_current)
        if self.solver =='rms':
            return optim.RMSprop(parameters, lr=self.learning_rate_current)
    def _get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'identity':
            return nn.Identity()
        elif self.activation =='softplus':
            return nn.Softplus()
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
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        loss = 0
        for higher, lower in pairs:
            # RankNet loss: max(0, margin - (score_higher - score_lower))
            diff = outputs[higher] - outputs[lower]
            loss += torch.max(torch.tensor(0.0, device=self.device), self.margin - diff)
            
        return loss / len(pairs)
    
    def _compute_validation_score(self, X_val, y_val):
        """Compute validation loss"""
        self.model_.eval()
        with torch.no_grad():
            val_outputs = self.model_(X_val)
            val_loss = self._pairwise_ranking_loss(val_outputs, y_val)
        self.model_.train()
        return -val_loss.item()  # Negative because higher score is better
    
    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
  
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for early stopping if needed
        if self.early_stopping and len(y) >= 32:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=self.validation_fraction, 
                random_state=self.random_state
            )
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
        else:
            X_train, y_train = X_scaled, y
        
        # Move tensors to the appropriate device
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
        
        self.model_ = self._build_model(X.shape[1]).to(self.device)
        #optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate_init)
        optimizer = self._get_optimizer(self.model_.parameters())
        
        # Setup for early stopping and convergence detection
        best_score = -np.inf
        best_epoch = 0
        no_improvement_count = 0
        self.n_iter_ = self.max_iter  # Will be updated if early stopping occurs
        
        # Calculate batch size
        n_samples = X_train.shape[0]
        if self.batch_size == 'auto':
            batch_size = min(64, n_samples)
        else:
            batch_size = self.batch_size
        
        indices = np.arange(n_samples)
        prev_loss = np.inf
        
        pbar = tqdm(range(self.max_iter), desc=f"Training on {self.device} batch size {batch_size}", leave=False,disable=not self.verbose)
        pbar.update(1)
        for epoch in pbar:
            np.random.shuffle(indices)
            total_loss = 0
            batches = 0

            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
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
                batches += 1
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / batches if batches > 0 else 0
            
            # Check for early stopping if enabled
            if self.early_stopping and len(y) >= 32:
                val_score = self._compute_validation_score(X_val_tensor, y_val_tensor)
                if val_score > best_score + self.tol:
                    best_score = val_score
                    best_epoch = epoch
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                if self.verbose:
                    pbar.set_postfix({

                        'train_loss': "{:.1e}".format(avg_loss), 
                        'val_score': val_score,
                        'no_improv': no_improvement_count
                    })
                
                if no_improvement_count >= self.n_iter_no_change:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}, best epoch was {best_epoch+1}")
                    self.n_iter_ = epoch + 1
                    break
            
            # Check for convergence based on training loss
            else:
                loss_improvement = prev_loss - avg_loss
                
                if self.verbose:
                    pbar.set_postfix({
                        'train_loss': "{:.1e}".format(avg_loss), 
                        'improvement': "{:.1e}".format(loss_improvement),
                        'no_improv': no_improvement_count if loss_improvement < self.tol else 0
                    })
                
                if loss_improvement < self.tol:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                
                if no_improvement_count >= self.n_iter_no_change:
                    if self.verbose:
                        pbar.set_postfix({"Converged at epoch":epoch+1})
                        pbar.update(self.max_iter)
                        pbar.close()
                    self.n_iter_ = epoch + 1
                    break
                
                prev_loss = avg_loss
        
        # Store final loss value
        self.loss_ = avg_loss
        
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model_.eval()
        with torch.no_grad():
            predictions = self.model_(X_tensor)
        
        # Move predictions back to CPU for numpy conversion
        return predictions.cpu().numpy().flatten()