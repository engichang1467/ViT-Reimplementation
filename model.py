import torch
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange
from einops.layers.torch import Rearrange


# Define the patch embedding layer
class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.projection = nn.Sequential(
            # Use a convolutional layer to extract patches and their embeddings simultaneously
            nn.Conv2d(
                config["channels"],
                config["dim"],
                kernel_size=config["batch_size"],
                stride=config["batch_size"],
            ),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["dim"]))
        self.positions = nn.Parameter(
            torch.randn(
                (config["image_size"] // config["batch_size"]) ** 2 + 1, config["dim"]
            )
        )

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positions
        return x


# Define the Multi-Head Attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = nn.MultiheadAttention(
            embed_dim=config["dim"],
            num_heads=config["heads"],
            dropout=config["dropout"],
        )
        self.norm = nn.LayerNorm(config["dim"])

    def forward(self, x):
        x = self.norm(x)
        x = self.attention(x, x, x)[0]
        return x


# Define the Feedforward layer
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config["dim"], config["mlp_dim"]),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["mlp_dim"], config["dim"]),
            nn.Dropout(config["dropout"]),
        )
        self.norm = nn.LayerNorm(config["dim"])

    def forward(self, x):
        x = self.norm(x)
        x = self.net(x)
        return x


# Define the Transformer Block combining MHA and Feedforward
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.feed_forward(x)
        return x


# Define the Vision Transformer Model
class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embedding = PatchEmbedding(config)
        self.transformer = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["depth"])]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config["dim"]), nn.Linear(config["dim"], config["num_classes"])
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0])
        return x
