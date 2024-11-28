import torch
import torch.nn as nn

from typing import Tuple, List, Dict, Any, Union, Optional

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super(LSTMEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        _, (h_n, _) = self.encoder(x)

        # get the hidden state from the last layer
        z = h_n[-1]

        return z


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, seq_len: int, num_layers: int = 1, dropout: float = 0.0):
        super(LSTMDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len

        self.decoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
    
    def forward(self, z: torch.tensor) -> torch.tensor:
        x_recon, (_, _) = self.decoder(z.unsqueeze(1).repeat(1, self.seq_len, 1))
    
        return x_recon
    

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_layers: int=1, dropout: float=0.0, seq_len: int=10):
        """
        LSTM Autoencoder model for sequence data.

        Args:
            input_dim (int): Number of features in input sequence
            hidden_dim (int): Hidden size of LSTM
            latent_dim (int): Latent space dimension
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            seq_len (int): Length of input sequence
        """
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len

        self.encoder = LSTMEncoder(input_dim=input_dim, hidden_dim=latent_dim, num_layers=num_layers, dropout=dropout)
        self.decoder = LSTMDecoder(input_dim=latent_dim, hidden_dim=input_dim, num_layers=num_layers, dropout=dropout, seq_len=seq_len,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: Reconstructed output sequence of shape (batch_size, seq_len, input_dim)
        """
        # Encode the input sequence
        z = self.encoder(x)

        # Decode the latent representation
        x_recon = self.decoder(z)

        return x_recon
    
        
def main():
    # Hyperparameters
    input_dim = 1   # Number of features in input sequence
    latent_dim = 16 # Latent space dimension
    seq_len = 10    # Length of input sequence
    batch_size = 1  # Number of sequences in a batch

    # Example input
    x = torch.randn(batch_size, seq_len, input_dim)  # (batch_size, seq_len, input_dim)

    # Initialize LSTM Autoencoder
    lstm_ae = LSTMAutoencoder(input_dim=input_dim, latent_dim=latent_dim, seq_len=seq_len)

    # Forward pass 
    x_recon = lstm_ae(x)

    # Reconstruction loss (MSE)
    criterion = nn.MSELoss()
    loss = criterion(x_recon, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_recon.shape}")
    print(f"Reconstruction Loss: {loss.item()}")


if __name__ == '__main__':
    main()