import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.misc import SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth

# TODO: Move MLP to a common layers package
from .vit import MLP


class MBConv(nn.Module):
    """
    Mobile inverted bottleneck convolution (MBConv) block from EfficientNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downsample (bool): Whether to downsample the spatial resolution.
        expansion_rate (int, optional): Expansion rate for the depthwise convolution. Defaults to 4.
        squeeze_rate (float, optional): Squeeze rate for the squeeze-and-excitation operation. Defaults to 0.25.
        stochastic_dropout (float, optional): Dropout rate for randomly dropping sample from block conv path. Defaults to 0.1.
    Shape:
        - Input: (B, in_channels, H, W)
        - Output: (B, out_channels, H // 2 if downsample else H, W // 2 if downsample else W)

    Examples:
        >>> x = torch.randn(4, 32, 224, 224)  # Batch size, channels, height, width
        >>> mbconv = MBConv(32, 64, True)
        >>> y = mbconv(x)
        >>> y.shape  # torch.Size([4, 64, 112, 112])
    """
    def __init__(
        self, in_channels, out_channels, downsample, expansion_rate=4, squeeze_rate=0.25, stochastic_dropout=0.1
    ):
        super().__init__()
        assert (expansion_rate > 0) and (squeeze_rate > 0)
        hidden_channels = int(expansion_rate * out_channels)
        squeeze_channels = int(squeeze_rate * hidden_channels)
        if downsample:
            stride = 2
        else:
            stride = 1

        self.block_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(
                in_channels=in_channels, out_channels=hidden_channels, kernel_size=1
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_channels,
            ),
            nn.BatchNorm2d(hidden_channels),
            SqueezeExcitation(
                input_channels=hidden_channels,
                squeeze_channels=squeeze_channels,
                activation=nn.SiLU,
            ),
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=out_channels, kernel_size=1
            ),
            # nn.BatchNorm2d(out_channels),
        )

        self.drop_path = StochasticDepth(stochastic_dropout, "row")
        if downsample:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1
                ),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        x = self.shortcut(x) + self.drop_path(self.block_conv(x))
        return x


class RelativeAttention(nn.Module):
    """
    Relative self-attention module for MaxViT.

    Args:
        embedding_dim (int): Embedding dimension.
        window_size (tuple, optional): Size of the attention window or grid. Defaults to (7, 7).
        num_heads (int, optional): Number of attention heads. Defaults to 32.
        dropout (float, optional): Dropout rate for attention and MLP weights. Defaults to 0.
    Shape:
        - Input: (B * num_windows, window_size[0] * window_size[1], C)
        - Output: (B * num_windows, window_size[0] * window_size[1], C)

    Examples:
        >>> x = torch.randn(4 * 49, 7 * 7, 96)  # Batch size * num_windows, height * width, channels
        >>> attention = RelativeAttention(96)
        >>> y = attention(x)
        >>> y.shape  # torch.Size([4 * 49, 7 * 7, 96])
    """
    def __init__(self, embedding_dim, window_size=(7, 7), num_heads=32, dropout=0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * self.num_heads == self.embedding_dim
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(self.embedding_dim, self.embedding_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            )
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.init_relative_position_index()

    def init_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):
        n_batch, window_len = x.shape[:2]
        qkv = (
            self.qkv(x)
            .reshape(n_batch, window_len, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ (k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        # Add relative position bias
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(n_batch, window_len, self.embedding_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionBlock(nn.Module):
    """
    Attention block in MaxViT, consisting of either window-based or relative self-attention.

    Args:
        embedding_dim (int): Embedding dimension.
        window_mode (bool, optional): Whether to use window-based attention or grid-based attention. Defaults to True.
        window_size (tuple, optional): Size of the attention window or grid. Defaults to (7, 7).
        num_heads (int, optional): Number of attention heads. Defaults to 32.
        dropout (float, optional): Dropout rate for attention weights. Defaults to 0.
        stochastic_dropout (float, optional): Dropout rate for randomly dropping sample from the window attention and mlp. Defaults to 0.1.
    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W)

    Examples:
        >>> x = torch.randn(4, 96, 56, 56)  # Batch size, channels, height, width
        >>> attention_block = AttentionBlock(96)
        >>> y = attention_block(x)
        >>> y.shape  # torch.Size([4, 96, 56, 56])
    """
    def __init__(
        self,
        embedding_dim,
        window_mode=True,
        window_size=(7, 7),
        num_heads=32,
        dropout=0,
        stochastic_dropout=0.1
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.window_mode = window_mode
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.drop_path = StochasticDepth(stochastic_dropout, "row")
        self.attn = RelativeAttention(
            embedding_dim=embedding_dim,
            window_size=window_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(
            embedding_dim=embedding_dim,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
        )

    def window_partition(self, x):
        """
        Args:
          x: (B, H, W, C)

        Returns:
          windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(
            B,
            H // self.window_size[0],
            self.window_size[0],
            W // self.window_size[1],
            self.window_size[1],
            C,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, self.window_size[0], self.window_size[1], C)
        )

        return windows

    def grid_partition(self, x):
        """
        Args:
          x: (B, H, W, C)

        Returns:
          windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(
            B,
            self.window_size[0],
            H // self.window_size[0],
            self.window_size[1],
            W // self.window_size[1],
            C,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, self.window_size[0], self.window_size[1], C)
        )

        return windows

    def window_reverse(self, windows, H: int, W: int):
        """
        Args:
          windows: (num_windows*B, window_size, window_size, C)
          H (int): Height of image
          W (int): Width of image

        Returns:
          x: (B, H, W, C)
        """
        C = windows.shape[-1]
        x = windows.view(
            -1,
            H // self.window_size[0],
            W // self.window_size[1],
            self.window_size[0],
            self.window_size[1],
            C,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        return x

    def grid_reverse(self, windows, H: int, W: int):
        """
        Args:
          windows: (num_windows*B, window_size, window_size, C)
          H (int): Height of image
          W (int): Width of image

        Returns:
          x: (B, H, W, C)
        """
        C = windows.shape[-1]
        x = windows.view(
            -1,
            H // self.window_size[0],
            W // self.window_size[1],
            self.window_size[0],
            self.window_size[1],
            C,
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, H, W, C)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # put channel last
        if self.window_mode:
            window = self.window_partition(x)
        else:
            window = self.grid_partition(x)

        window = window.view(-1, self.window_size[0] * self.window_size[1], C)
        window = self.drop_path(self.attn(self.norm1(window)))

        if self.window_mode:
            window = self.window_reverse(window, H, W)
        else:
            window = self.grid_reverse(window, H, W)

        x = x + window
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, 3, 1, 2)  # put channel at position 1

        return x


class MaxViTBlock(nn.Module):
    """
    MaxViT block, combining MBConv and attention modules.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downsample (bool, optional): Whether to downsample the spatial resolution. Defaults to False.
        window_size (tuple, optional): Size of the attention window or grid. Defaults to (7, 7).
        window_modes (list, optional): List of window or grid modes for each attention block. Defaults to [True, False].
        num_heads (int, optional): Number of attention heads. Defaults to 32.
        dropout (float, optional): Dropout rate for attention weights. Defaults to 0.
    Shape:
        - Input: (B, in_channels, H, W)
        - Output: (B, out_channels, H // 2 if downsample else H, W // 2 if downsample else W)

    Example:
        >>> block = MaxViTBlock(64, 96, downsample=True, window_size=(7, 7), window_modes=[True, False], num_heads=32)
        >>> x = torch.randn(4, 64, 224, 224)  # Batch size, channels, height, width
        >>> y = block(x)
        >>> y.shape  # torch.Size([4, 96, 112, 112])
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        downsample=False,
        window_size=(7, 7),
        window_modes=[True, False],
        num_heads=32,
        dropout=0,
        stochastic_dropout=0.1
    ):
        super().__init__()
        layers = nn.ModuleList([])
        conv_block = MBConv(in_channels, out_channels, downsample, stochastic_dropout=stochastic_dropout)

        layers.append(conv_block)
        layers.extend(
            [
                AttentionBlock(
                    out_channels,
                    window_mode=mode,
                    window_size=window_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    stochastic_dropout=stochastic_dropout
                )
                for mode in window_modes
            ],
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class MaxVit(nn.Module):
    """
    MaxViT architecture for image classification, combining MBConv and window-based/relative self-attention.

    Args:
        output_dim (int, optional): Number of output classes. Defaults to 10.
        embedding_dims (list, optional): List of embedding dimensions for each stage. Defaults to [64, 96, 192, 384, 768].
        window_size (tuple, optional): Size of the attention window. Defaults to (7, 7).
        block_depths (list, optional): Number of MaxViTBlocks in each stage. Defaults to [2, 6, 14, 2].
        num_heads (list, optional): Number of attention heads in each stage. Defaults to [32, 32, 32, 32].
        dropout (float, optional): Dropout rate for MLP layers. Defaults to 0.0.
        stochastic_dropout (float, optional): Dropout rate for randomly dropping sample from the Attention or MBConv block. Defaults to 0.1.
    Shape:
        - Input: (B, 3, H, W)
        - Output: (B, output_dim)

    Example:
        >>> model = MaxViT(output_dim=1000)
        >>> x = torch.randn(4, 3, 224, 224)  # Batch size, channels, height, width
        >>> logits = model(x)
        >>> logits.shape  # torch.Size([4, 1000])
    """
    def __init__(
        self,
        output_dim=10,
        embedding_dims=[64, 96, 192, 384, 768],
        window_size=(7, 7),
        block_depths=[2, 6, 14, 2],
        num_heads=[32, 32, 32, 32],
        dropout=0.0,
        stochastic_dropout=0.1
    ):
        super().__init__()
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

        self.layers.append(stem)
        in_channels = stem_channels
        sum_block_depths = sum(block_depths)
        layer_counter = 0
        for idx_block, block_depth in enumerate(block_depths):
            downsample = True  # new block
            out_channels = embedding_dims[
                idx_block + 1
            ]  # first index is for stem channels
            layers = nn.ModuleList([])
            for _ in range(block_depth):
                stochastic_layer_dropout = stochastic_dropout * layer_counter / (sum_block_depths - 1)
                layer_counter += 1
                block = MaxViTBlock(
                    in_channels,
                    out_channels,
                    downsample,
                    window_size=window_size,
                    num_heads=num_heads[idx_block],
                    dropout=dropout,
                    stochastic_dropout=stochastic_layer_dropout,
                )

                layers.append(block)
                downsample = False
                in_channels = out_channels
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
