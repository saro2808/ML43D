import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, input_dim, num_freqs):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs

    def forward(self, x):
        """
        Compute the positional encoding of the input.
        Output = [x, sin(2^0 * x), cos(2^0 * x), sin(2^1 * x), cos(2^1 * x), ..., sin(2^num_freqs * x), cos(2^num_freqs * x]
        Args:
            x: (N, input_dim)
        Returns:
            (N, input_dim * (num_freqs * 2 + 1))
        """
        N, D = x.shape
        freq_bands = 2 ** torch.arange(self.num_freqs, device=x.device)  # (num_freqs,)

        # x: (N, D) -> (N, D, num_freqs)
        x_expanded = x[..., None]
        x_freq = x_expanded * freq_bands
        sin_enc = torch.sin(x_freq)  # (N, D, num_freqs)
        cos_enc = torch.cos(x_freq)  # (N, D, num_freqs)

        # Stack as: (N, D, num_freqs, 2) where [:,:,:,0] = sin, [:,:,:,1] = cos
        pe = torch.stack([sin_enc, cos_enc], dim=-1)
    
        # Interleave sin and cos per frequency → reshape (N, D, num_freqs * 2)
        pe = pe.reshape(N, D, -1)
        
        return torch.cat([x_expanded, pe], dim=-1).reshape(N, -1)


class SDFField(nn.Module):

    def __init__(self):
        super().__init__()
        self.activation = nn.Softplus
        self.embed_fn = PositionalEmbedding(3, 6)
        # Define the rest of the model
        self.net1 = nn.Sequential(
            nn.Linear(39, 128),
            self.activation(beta=100),
            nn.Linear(128, 128),
            self.activation(beta=100),
            nn.Linear(128, 128),
            self.activation(beta=100),
            nn.Linear(128, 89),
            self.activation(beta=100),
        )
        self.net2 = nn.Sequential(
            nn.Linear(128, 128),
            self.activation(beta=100),
            nn.Linear(128, 128),
            self.activation(beta=100),
            nn.Linear(128, 128),
            self.activation(beta=100),
            nn.Linear(128, 129)
        )

    def forward(self, x):
        """
        Args:
            x: (N, 3) input points
        Output:
            (N, 1 + latent_size) tensor. The first value is the SDF value and the rest are the latent code.
        """
        # implement forward pass
        # The forward pass should contain the following steps:
        # 1. Compute the positional encoding of the input
        # 2. Apply mlp layers. Add skip connections if needed.
        pe = self.embed_fn(x)
        x1 = self.net1(pe)
        return self.net2(torch.cat([pe, x1], dim=1))

    def get_sdf(self, x):
        """Get the SDF value only without the latent code"""
        sdf = self.forward(x)[:, :1]
        return sdf

    def gradient(self, x):
        """Compute the normal direction using the gradient of the SDF"""
        x.requires_grad_(True)
        y = self.get_sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class ColorField(nn.Module):

    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.embed_fn = PositionalEmbedding(3, 4)
        # Define the rest of the model
        self.net = nn.Sequential(
            nn.Linear(161, 128),
            self.activation,
            nn.Linear(128, 128),
            self.activation,
            nn.Linear(128, 128),
            self.activation,
            nn.Linear(128, 128),
            self.activation,
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, xyz, normals, view_dirs, features):
        """
        Args:
            xyz: (N, 3) input points
            normals: (N, 3) input normals
            view_dirs: (N, 3) input view directions
            features: (N, C) input features
        Returns:
            (N, 3) color output
        """
        # implement forward pass
        # The forward pass should contain the following steps:
        # 1. Add positional encoding to the view_dirs
        # 2. Concatenate the input features
        # 3. Apply mlp layers
        # 4. Apply sigmoid on the color output
        pe = self.embed_fn(view_dirs)
        return self.net(torch.cat([xyz, normals, features, pe], dim=1))


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val=0.3):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)


def compute_psnr(x, y, mask=None):
    # implement PSNR computation
    valid_pix_num = x.numel()
    diff = (x - y)
    if mask is not None:
        diff *= mask
        valid_pix_num = mask.sum()
    mse = torch.sum(diff**2) / valid_pix_num / 3
    return 10 * torch.log10(1 / (mse + 1e-8))
