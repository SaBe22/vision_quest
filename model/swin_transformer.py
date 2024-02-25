import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.stochastic_depth import StochasticDepth
# TODO: Move PatchEmbedding and MLP to a common layers package
from model.vit import PatchEmbedding, MLP


class PatchMerging(nn.Module):
    """
    Patch merging layer in Swin Transformer for dimensionality reduction.

    Args:
        embedding_dim (int): Input and output embedding dimension.
        pre_norm (bool, optional): Apply layer normalization before the merging operation. Defaults to True.

    Shape:
        - Input: (B, H, W, C)
        - Output: (B, H // 2, W // 2, embedding_dim * 2)

    Examples:
        >>> patch_merging = PatchMerging(embedding_dim=128)
        >>> x = torch.randn(4, 16, 16, 128)  # Batch size, height, width, channels
        >>> y = patch_merging(x)
        >>> y.shape  # torch.Size([4, 8, 8, 256])
    """
    def __init__(self, embedding_dim, pre_norm=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pre_norm = pre_norm
        if self.pre_norm:
            self.norm = nn.LayerNorm(4 * embedding_dim)
        self.reduction = nn.Linear(4 * embedding_dim, 2 * embedding_dim)
        if not self.pre_norm:
            self.norm = nn.LayerNorm(2 * embedding_dim)

    def forward(self, x):
        n_batch, height, width = x.shape[:3]
        assert height % 2 == 0 and width % 2 == 0
        # assert height * width == window_len
        x = x.view(n_batch, height, width, self.embedding_dim)
        x = x.reshape(n_batch, height // 2, 2, width // 2, 2, self.embedding_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).flatten(
            start_dim=3
        )  # n_batch x height // 2 x width // 2 x 2 * 2 * embedding_dim
        if self.pre_norm:
            x = self.norm(x)
        x = self.reduction(x)
        if not self.pre_norm:
            x = self.norm(x)

        return x


class WindowAttention(nn.Module):
    """
    Window-based self-attention layer in Swin Transformer.

    Args:
        embedding_dim (int): Embedding dimension.
        window_size (tuple): Size of the attention window.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Dropout rate for attention and projection weights. Defaults to 0.0.

    Shape:
        - Input: (B * num_windows, window_size[0] * window_size[1], embedding_dim)
        - Output: (B * num_windows, window_size[0] * window_size[1], embedding_dim)

    Examples:
        >>> window_attention = WindowAttention(embedding_dim=128, window_size=(7, 7), num_heads=8)
        >>> x = torch.randn(4 * 49, 7 * 7, 128)
        >>> y = window_attention(x)
        >>> y.shape  # torch.Size([4 * 49, 7 * 7, 128])
    """

    def __init__(self, embedding_dim, window_size, num_heads, dropout=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size  # Wh, Ww
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
        self.init_relative_position_index()
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def init_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
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

    def forward(self, x, attn_mask=None):
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

        if attn_mask is not None:
            n_window = attn_mask.shape[0]
            attn = attn.view(
                n_batch // n_window, n_window, self.num_heads, window_len, window_len
            ) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, window_len, window_len)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(n_batch, window_len, self.embedding_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):
    """
    Swin Transformer block, consisting of PatchMerging, WindowAttention, and MLP layers.

    Args:
        embedding_dim (int): Input and output embedding dimension.
        num_heads (int): Number of attention heads in the WindowAttention layer.
        patch_shape (tuple): Shape of the patches (height, width).
        window_size (tuple): Size of the attention window.
        shift_size (int, optional): Shift size for overlapping window partition in downsampling layers. Defaults to None.
        dropout (float, optional): Dropout rate for the window attention and mlp layer. Defaults to 0.
        stochastic_dropout (float, optional): Dropout rate for randomly dropping sample from the window attention and mlp. Defaults to 0.1.

    Shape:
        - Input: (B, H, W, C)
        - Output: (B, H, , C)

    Examples:
        >>> swin_block = SwinBlock(embedding_dim=128, num_heads=8, patch_shape=(4, 4), window_size=(7, 7))
        >>> x = torch.randn(4, 16, 16, 128)  # Batch size, height, width, channels
        >>> y = swin_block(x)
        >>> y.shape  # torch.Size([4, 16, 16, 128])
    """
    def __init__(
        self,
        embedding_dim,
        num_heads,
        patch_shape,
        window_size,
        shift_size=None,
        dropout=0,
        stochastic_dropout=0.1
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.patch_shape = patch_shape
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.window_attention = WindowAttention(
            embedding_dim=embedding_dim, num_heads=num_heads, window_size=window_size, dropout=dropout
        )

        self.path_dropout = StochasticDepth(stochastic_dropout, "row")

        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(
            embedding_dim=embedding_dim,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
        )

        if self.shift_size is not None:
            assert (0 <= self.shift_size[0] < self.window_size[0]) and (
                0 <= self.shift_size[1] < self.window_size[1]
            )

            # calculate attention mask for SW-MSA
            H, W = self.patch_shape
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.shift_size[0]),
                slice(-self.shift_size[0], None),
            )
            w_slices = (
                slice(0, -self.window_size[1]),
                slice(-self.window_size[1], -self.shift_size[1]),
                slice(-self.shift_size[1], None),
            )
            count = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = count
                    count += 1

            mask_windows = self.window_partition(
                img_mask
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(
                -1, self.window_size[0] * self.window_size[1]
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask, persistent=False)

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

    def window_reverse(self, windows, H: int, W: int):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
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

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.norm1(x)
        if self.shift_size is not None:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)
            )
            # partition windows
            x_windows = self.window_partition(
                shifted_x
            )  # nW*B, window_size, window_size, C
        else:
            shifted_x = x
            # partition windows
            x_windows = self.window_partition(
                shifted_x
            )  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(
            -1, self.window_size[0] * self.window_size[1], C
        )  # nW*B, window_size*window_size, C

        attn_window = self.window_attention(x_windows, attn_mask=self.attn_mask)

        # merge windows
        attn_window = attn_window.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = self.window_reverse(attn_window, H, W)  # B H W C

        # reverse cyclic shift
        if self.shift_size is not None:
            shifted_x = torch.roll(
                shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2)
            )

        x = x + self.path_dropout(shifted_x)
        x = x + self.path_dropout(self.mlp(self.norm2(x)))
        return x


class SwinTransformer(nn.Module):
    """
    Swin Transformer architecture for image classification.

    Args:
        embedding_dim (int, optional): Input and output embedding dimension. Defaults to 96.
        img_size (tuple, optional): Input image size (height, width). Defaults to (224, 224).
        patch_size (tuple, optional): Size of the patches (height, width). Defaults to (4, 4).
        output_dim (int, optional): Number of output classes. Defaults to 10.
        block_depths (list, optional): Number of SwinBlocks in each stage. Defaults to [2, 2, 6, 2].
        num_heads (list, optional): Number of attention heads in each stage. Defaults to [3, 6, 12, 24].
        window_size (tuple, optional): Size of the attention window. Defaults to (7, 7).
        dropout (float, optional): Dropout rate for MLP layers. Defaults to 0.
        stochastic_dropout (float, optional): Dropout rate for randomly dropping sample from the window attention and mlp. Defaults to 0.1.

    Shape:
        - Input: (B, img_size[0], img_size[1], 3)
        - Output: (B, output_dim)

    Examples:
        >>> swin_transformer = SwinTransformer(embedding_dim=128, num_heads=[4, 8, 16], output_dim=10)
        >>> x = torch.randn(4, 224, 224, 3)  # Batch size, height, width, channels
        >>> logits = swin_transformer(x)
        >>> logits.shape  # torch.Size([4, 10])
    """
    def __init__(
        self,
        embedding_dim=96,
        img_size=(224, 224),
        patch_size=(4, 4),
        output_dim=10,
        block_depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(7, 7),
        dropout=0,
        stochastic_dropout=0.1
    ):
        super().__init__()
        assert len(block_depths) == len(num_heads)

        self.patch_embedding = PatchEmbedding(
            embedding_dim=embedding_dim, patch_sizes=patch_size, img_sizes=img_size
        )
        patch_shape = self.patch_embedding.patch_shape.copy()
        sum_block_depths = sum(block_depths)
        layer_counter = 0
        self.swin_stage_layers = nn.ModuleList()
        for idx_block, block_depth in enumerate(block_depths):
            if idx_block > 0:
                patch_merging = PatchMerging(embedding_dim=embedding_dim)
                self.swin_stage_layers.append(patch_merging)
                # After each block embedding is multiplied by 2
                embedding_dim *= 2

            for n_layer in range(block_depth):
                shift_size = (
                    (window_size[0] // 2, window_size[1] // 2) if n_layer % 2 == 1 else None
                )
                stochastic_layer_dropout = stochastic_dropout * layer_counter / (sum_block_depths - 1)
                block = SwinBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads[idx_block],
                    patch_shape=patch_shape,
                    window_size=window_size,
                    shift_size=shift_size,
                    dropout=dropout,
                    stochastic_dropout=stochastic_layer_dropout,
                )
                self.swin_stage_layers.append(block)

            # After each block resolution is downsampled by 2
            patch_shape[0] = patch_shape[0] // 2
            patch_shape[1] = patch_shape[1] // 2

        self.norm = nn.LayerNorm(embedding_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(embedding_dim, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        w = self.patch_embedding.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        n_batch = x.shape[0]
        x = self.patch_embedding(x)
        x = x.view(
            n_batch,
            self.patch_embedding.patch_shape[0],
            self.patch_embedding.patch_shape[1],
            -1,
        )
        for layer in self.swin_stage_layers:
            x = layer(x)

        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.avg_pool(x).flatten(1)
        return self.head(x)
