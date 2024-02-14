import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Embeds image patches into lower-dimensional vectors.

    This module performs 2D convolutional embedding to transform image patches
    into lower-dimensional vectors suitable for the Vision Transformer architecture.

    Args:
        embedding_dim (int): The output dimension of the embedded vectors.
        patch_sizes (tuple of int): Tuple specifying the patch sizes in the height
            and width dimensions.
        img_sizes (tuple of int): Tuple specifying the input image sizes in the
            height and width dimensions.

    Attributes:
        patch_embed (nn.Conv2d): The 2D convolutional layer for embedding patches.
    """
    def __init__(self, embedding_dim, patch_sizes, img_sizes):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.patch_sizes = patch_sizes
        self.img_sizes = img_sizes

        self.n_patches = (img_sizes[0] // patch_sizes[0]) * (img_sizes[1] // patch_sizes[1])
        self.proj = nn.Conv2d(in_channels=3, out_channels=embedding_dim, kernel_size=patch_sizes, stride=patch_sizes)

    def forward(self, x):
        """Computes the patch embedding of the input image.

        Args:
            x (torch.Tensor): The input image tensor of shape (B, C, H, W),
                where B is batch size, C is number of channels (typically 3),
                H is height, and W is width.

        Returns:
            torch.Tensor: The embedded patches tensor of shape (B, N, embedding_dim),
                where N is the total number of patches.
        """
        x = self.proj(x)
        x = x.reshape(-1, self.embedding_dim, self.n_patches)
        x = x.permute(0, 2, 1)
        return x
    
class MLP(nn.Module):
    """Multilayer perceptron (MLP) with linear layers and activation functions.

    This module implements a feed-forward MLP with a linear layer, a ReLU
    activation, and another linear layer with dropout regularization.

    Args:
        embedding_dim (int): The embedding dimension of the input and output.
        dim_feedforward (int): The intermediate dimension of the MLP.
        dropout (float): The dropout probability to apply after the ReLU activation.
    """
    def __init__(self, embedding_dim, dim_feedforward, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, dim_feedforward)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim_feedforward, embedding_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
class BlockVisionTransformer(nn.Module):
    def __init__(self, embedding_dim, dim_feedforward, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, dim_feedforward, dropout)       

    def forward(self, x):
        """Computes the forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor of shape (B, N, embedding_dim),
                where B is batch size, N is number of elements, and embedding_dim
                is the embedding dimension.

        Returns:
            torch.Tensor: The output tensor of the MLP, with the same shape as
                the input tensor.
        """
        x_att = self.norm1(x)
        x_att, _ = self.attention(x_att, x_att, x_att, need_weights=False)
        x = x + x_att
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer model for image classification or other vision tasks.

    This module implements the Vision Transformer architecture with multiple
    Transformer encoder blocks, a positional encoding layer, and a final
    classification head.

    Args:
        embedding_dim (int, optional): The embedding dimension of the input and output. Defaults to 768.
        output_dim (int, optional): The number of output classes for classification. Defaults to 64.
        patch_sizes (tuple of int, optional): Tuple specifying the patch sizes in the height and width dimensions. Defaults to (16, 8).
        img_sizes (tuple of int, optional): Tuple specifying the input image sizes in the height and width dimensions. Defaults to (128, 64).
        num_layers (int, optional): The number of transformer encoder blocks. Defaults to 12.
        num_heads (int, optional): The number of attention heads in each transformer block. Defaults to 12.
        dim_feedforward (int, optional): The intermediate dimension of the MLP in each transformer block. Defaults to 3072.
        dropout (float, optional): The dropout probability to apply after various layers. Defaults to 0.1.

    Attributes:
        patch_embed (PatchEmbedding): The patch embedding module.
        pos_embed (nn.Parameter): The positional encoding tensor.
        cls_token (nn.Parameter): The class token tensor.
        blocks (nn.ModuleList): A list of Transformer encoder blocks.
        head (nn.Linear): The final classification layer.
    """
    def __init__(self, embedding_dim=768, output_dim=64, patch_sizes=(16, 8), img_sizes=(128, 64), num_layers=12, num_heads=12, dim_feedforward=3072, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(embedding_dim, patch_sizes, img_sizes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embed = nn.Parameter(
                torch.randn(1, 1 + self.patch_embedding.n_patches, embedding_dim), # + 1 cause of class token
        )
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList(
            [
                BlockVisionTransformer(
                    embedding_dim=embedding_dim, dim_feedforward=dim_feedforward, num_heads=num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        w = self.patch_embedding.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
               nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)        
        

    def forward(self, x):
        n_batch = x.shape[0]
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(n_batch, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)
