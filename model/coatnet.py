import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.stochastic_depth import StochasticDepth

# TODO: Merge maxvit and coatnet
from .maxvit import RelativeAttention, MBConv
# TODO: Move MLP to a common layers package
from .vit import MLP


class CoAtAttention(nn.Module):
    """
    CoAtNet relative self-attention layer.

    Args:
        embedding_dim (int): Embedding dimension.
        img_size (tuple, optional): Input image size (height, width). Defaults to (16, 16).
        num_heads (int, optional): Number of attention heads. Defaults to 32.
        dropout (float, optional): Dropout rate for attention weights. Defaults to 0.0.
        stochastic_dropout (float, optional): Dropout rate for randomly dropping channels in channel attention branch. Defaults to 0.1.

    Shape:
        - Input: (B, C, img_size[0], img_size[1])
        - Output: (B, C, img_size[0], img_size[1])

    Examples:
        >>> attention = CoAtAttention(embedding_dim=128, num_heads=8)
        >>> x = torch.randn(4, 128, 16, 16)  # Batch size, channels, height, width, 
        >>> y = attention(x)
        >>> y.shape  # torch.Size([4, 128, 16, 16])
    """
    def __init__(self, embedding_dim, img_size=(16, 16), num_heads=32, dropout=0.0, stochastic_dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.drop_path = StochasticDepth(stochastic_dropout, "row")
        self.rel_attn = RelativeAttention(embedding_dim=embedding_dim, window_size=img_size, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(
            embedding_dim=embedding_dim,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # put channel last
        x = x.view(-1, H * W, C)
        x = x + self.drop_path(self.rel_attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

class CoAtNet(nn.Module):
    """
    CoAtNet architecture for image classification, combining MBConv and relative self attention.

    Args:
        img_size (tuple, optional): Input image size (height, width). Defaults to (224, 224).
        output_dim (int, optional): Number of output classes. Defaults to 10.
        embedding_dims (list, optional): List of embedding dimensions for each stage. Defaults to [64, 96, 192, 384, 768].
        block_depths (list, optional): Number of blocks in each stage. Defaults to [2, 6, 14, 2].
        block_types (list, optional): List of block types for each stage ("C" for MBConv block, "T" for relative self-attention). Defaults to ["C", "C", "T", "T"].
        num_heads (int, optional): Number of attention heads. Defaults to 32.
        dropout (float, optional): Dropout rate for MLP layers. Defaults to 0.0.
        stochastic_dropout (float, optional): Dropout rate for randomly dropping channels in channel attention branch. Defaults to 0.1.

    Shape:
        - Input: (B, 3, img_size[0], img_size[1])
        - Output: (B, output_dim)

    Examples:
        >>> model = CoAtNet(output_dim=1000)
        >>> x = torch.randn(4, 3, 224, 224)  # Batch size, channels, height, width
        >>> logits = model(x)
        >>> logits.shape  # torch.Size([4, 1000])
    """

    def __init__(self, img_size=(224, 224), output_dim=10, embedding_dims=[64, 96, 192, 384, 768], block_depths=[2, 6, 14, 2], block_types=["C", "C", "T", "T"], num_heads=32, dropout=0.0, stochastic_dropout=0.1):
        super().__init__()
        assert len(block_depths) == len(block_types)
        assert len(embedding_dims) == (len(block_depths) + 1)
        height, width = img_size
        self.layers = nn.ModuleList([])
        stem_channels = embedding_dims[0]
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels=stem_channels,
                out_channels=stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(stem_channels),
        )

        height = height // 2
        width = width // 2

        self.layers.append(stem)
        in_channels = stem_channels
        sum_block_depths = sum(block_depths)
        layer_counter = 0
        for idx_block, (block_type, block_depth) in enumerate(zip(block_types, block_depths)):
            downsample = True  # new block
            out_channels = embedding_dims[idx_block + 1]  # first index is for stem channels
            height = height // 2
            width = width // 2
            layers = nn.ModuleList([])
            for _ in range(block_depth):
                stochastic_layer_dropout = stochastic_dropout * layer_counter / (sum_block_depths - 1)
                layer_counter += 1
                if block_type == "C":
                    block = MBConv(in_channels, out_channels, downsample, stochastic_dropout=stochastic_layer_dropout)
                elif block_type == "T":
                    if downsample:                      
                        layers.append(nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)))
                        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0))
                    block = CoAtAttention(embedding_dim=out_channels, img_size=(height, width), num_heads=num_heads, dropout=dropout, stochastic_dropout=stochastic_layer_dropout)

                downsample = False
                in_channels = out_channels
                layers.append(block)

            self.layers.append(nn.Sequential(*layers))

        self.norm = nn.LayerNorm(in_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(in_channels, output_dim)
        self._init_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.avg_pool(x).flatten(1)
        x = self.norm(x)
        return self.head(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
