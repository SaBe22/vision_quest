import torch
import torch.nn as nn
from .vit import BlockVisionTransformer, PatchEmbedding

class VisionTransformerDecoder(nn.Module):
    """Decoder module for a Vision Transformer-based Masked Autoencoder.

    This module reconstructs masked image patches based on encoded representations
    from the encoder. It leverages multi-head attention and feed-forward layers
    to progressively reconstruct masked patches in a masked autoregressive manner.

    Args:
        patch_sizes (tuple of int, optional): Tuple specifying patch sizes.
            Defaults to (16, 16).
        img_sizes (tuple of int, optional): Tuple specifying input image sizes.
            Defaults to (224, 224).
        encoder_embedding_dim (int, optional): Embedding dimension of encoder outputs.
            Defaults to 768.
        decoder_embedding_dim (int, optional): Embedding dimension of decoder inputs.
            Defaults to 768.
        decoder_num_layers (int, optional): Number of decoder transformer blocks.
            Defaults to 12.
        decoder_num_heads (int, optional): Number of attention heads in decoder blocks.
            Defaults to 12.
        decoder_dim_feedforward (int, optional): Intermediate dimension of MLP in decoder blocks.
            Defaults to 3072.
        decoder_dropout (float, optional): Dropout probability in decoder blocks.
            Defaults to 0.1.

    Attributes:
        pos_embed (nn.Parameter): Learnable positional encoding for reconstructed patches.
        blocks (nn.ModuleList): List of decoder transformer blocks.
        head (nn.Linear): Final linear layer to project decoder outputs to pixel space.
    """
    def __init__(self, patch_sizes=(16, 16), img_sizes=(224, 224),
                 encoder_embedding_dim=768, decoder_embedding_dim=768,
                 decoder_num_layers=12, decoder_num_heads=12, decoder_dim_feedforward=3072, decoder_dropout=0.1):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.img_sizes = img_sizes
        self.num_patches = (img_sizes[0] // patch_sizes[0]) * (img_sizes[1] // patch_sizes[1])
        self.embedding = nn.Linear(encoder_embedding_dim, decoder_embedding_dim, bias=True)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_embedding_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, decoder_embedding_dim),
        ) 

        self.blocks = nn.ModuleList(
            [BlockVisionTransformer(decoder_embedding_dim, decoder_dim_feedforward, decoder_num_heads, decoder_dropout) for _ in range(decoder_num_layers)]
        )
        self.norm = nn.LayerNorm(decoder_embedding_dim)
        self.head = nn.Linear(decoder_embedding_dim, patch_sizes[0] * patch_sizes[1] * 3) # decoder to patch          
        
        self.initialize_weights()
    
    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

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

    def forward(self, x, indices_order_to_restore):
        """Reconstructs masked patches from encoded representations.

        Args:
            x (torch.Tensor): Encoded representations from the encoder.
            indices_order_to_restore (torch.Tensor): LongTensor containing masked patch indices in original order.

        Returns:
            torch.Tensor: Reconstructed image patches.
        """
        # embed tokens
        x = self.embedding(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], indices_order_to_restore.shape[1] + 1 - x.shape[1], 1)
        cls_token = x[:, :1, :]
        x_no_cls_token = x[:, 1:, :]
        x_ = torch.cat([x_no_cls_token, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=indices_order_to_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([cls_token, x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # predictor projection
        x = self.head(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

class VisionTransformerEncoder(nn.Module):
    """Encoder module for a Vision Transformer-based Masked Autoencoder.

    This module extracts visual features from image patches and applies random masking.
    It utilizes a standard Vision Transformer architecture with patch embedding,
    positional encoding, and transformer encoder blocks.

    Args:
        patch_sizes (tuple of int, optional): Tuple specifying patch sizes.
            Defaults to (16, 16).
        img_sizes (tuple of int, optional): Tuple specifying input image sizes.
            Defaults to (224, 224).
        embedding_dim (int, optional): Embedding dimension of encoder.
            Defaults to 768.
        num_layers (int, optional): Number of encoder transformer blocks.
            Defaults to 12.
        num_heads (int, optional): Number of attention heads in encoder blocks.
            Defaults to 12.
        dim_feedforward (int, optional): Intermediate dimension of MLP in encoder blocks.
            Defaults to 3072.
        dropout (float, optional): Dropout probability in encoder blocks.
            Defaults to 0.1.

    Attributes:
        patch_embed (PatchEmbedding): Patch embedding module to embed reconstructed patches.
        pos_embed (nn.Parameter): Learnable positional encoding for reconstructed patches.
        transformer_blocks (nn.ModuleList): List of decoder transformer blocks.
    """
    def __init__(self, patch_sizes=(16, 16), img_sizes=(224, 224), mask_ratio=0.75, embedding_dim=768, num_layers=12, num_heads=12, dim_feedforward=3072, dropout=0.1):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_embedding = PatchEmbedding(embedding_dim, patch_sizes, img_sizes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embed = nn.Parameter(
                torch.randn(1, 1 + self.patch_embedding.n_patches, embedding_dim), # + 1 cause of class token
        )

        self.blocks = nn.ModuleList(
            [
                BlockVisionTransformer(
                    embedding_dim=embedding_dim, dim_feedforward=dim_feedforward, num_heads=num_heads, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
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
    
    def random_mask(self, x):
        """Performs random masking of image patches with a specified ratio.

        This function randomly masks a given proportion of image patches within the input tensor.
        It creates a binary mask where 1 indicates a masked patch and 0 indicates a kept patch.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C, H, W), where B is batch size,
                            N is number of patches, C is number of channels (e.g., 3),
                            H is patch height, and W is patch width.

        Returns:
            torch.Tensor: A binary mask tensor of shape (B, N) indicating which patches
                        are masked (1) and which are kept (0).
        """
        n_batch, num_patches, embedding_dim = x.shape
        noise = torch.zeros(n_batch, num_patches).uniform_()
        indices = torch.argsort(noise, dim=1).to(x.device)  # ascend: small is keep, large is remove
        indices_order_to_restore = torch.argsort(indices, dim=1).to(x.device)
        length_to_keep = int(num_patches * (1 - self.mask_ratio))
        indices_to_keep = indices[:, :length_to_keep]
        x_masked = torch.gather(x, dim=1, index=indices_to_keep.unsqueeze(-1).repeat(1, 1, embedding_dim))

        mask = torch.ones([n_batch, num_patches], device=x.device)
        mask[:, :length_to_keep] = 0

        mask = torch.gather(mask, dim=1, index=indices_order_to_restore)

        return x_masked, mask, indices_order_to_restore

    def forward(self, x):
        """Extracts features and applies random masking.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Encoded representations of masked patches.
        """
        n_batch = x.shape[0]
        x = self.patch_embedding(x)
        x = x + self.pos_embed[:,1:] # 1st indice is for cls token position
        x, mask, indices_order_to_restore = self.random_mask(x)

        cls_token = self.cls_token.expand(n_batch, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed[:,:1]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x, mask, indices_order_to_restore

class MAEVisionTransformer(nn.Module):
    """Masked Autoencoder (MAE) model based on Vision Transformers.

    This model implements the MAE architecture, combining a Vision Transformer encoder
    for feature extraction and masking with a Vision Transformer decoder for patch reconstruction.
    """
    def __init__(self, patch_sizes=(16, 16), img_sizes=(224, 224), mask_ratio=0.75,
                 encoder_embedding_dim=768, encoder_num_layers=12, encoder_num_heads=12, encoder_dim_feedforward=3072, encoder_dropout=0.1,
                 decoder_embedding_dim=512, decoder_num_layers=8, decoder_num_heads=16, decoder_dim_feedforward=2048, decoder_dropout=0.1
    ):
        super().__init__()

        self.encoder = VisionTransformerEncoder(
            patch_sizes=patch_sizes, img_sizes=img_sizes, mask_ratio=mask_ratio,
            embedding_dim=encoder_embedding_dim, num_layers=encoder_num_layers, num_heads=encoder_num_heads,
            dim_feedforward=encoder_dim_feedforward, dropout=encoder_dropout,
        )

        self.decoder = VisionTransformerDecoder(
            patch_sizes=patch_sizes, img_sizes=img_sizes,
            encoder_embedding_dim=encoder_embedding_dim, decoder_embedding_dim=decoder_embedding_dim, decoder_num_layers=decoder_num_layers,
            decoder_num_heads=decoder_num_heads, decoder_dim_feedforward=decoder_dim_feedforward, decoder_dropout=decoder_dropout
        )

    def encode(self, x):
        x, mask, indices_order_to_restore = self.encoder(x)
        return x, mask, indices_order_to_restore

    def decode(self, x, indices_order_to_restore):
        x = self.decoder(x, indices_order_to_restore)
        return x

    def forward(self, x):
        """Performs masked autoencoding.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Reconstructed image tensor.
        """
        x, mask, indices_order_to_restore = self.encode(x)
        x = self.decode(x, indices_order_to_restore)
        return x, mask

class VITMAELoss(nn.Module):
    def __init__(self, patch_sizes=(16, 16), img_sizes=(224, 224)):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.num_patches_height = img_sizes[0] // self.patch_sizes[0]
        self.num_patches_width = img_sizes[1] // self.patch_sizes[1]

    def forward(self, x, mask, target):
        n_batch, n_channels = target.shape[:2]
        target = target.reshape(n_batch, n_channels, self.num_patches_height, self.patch_sizes[0], self.num_patches_width, self.patch_sizes[1])
        target = torch.einsum("bchpwq -> bhwpqc", target)
        target = target.reshape(n_batch, self.num_patches_height * self.num_patches_width, self.patch_sizes[0] * self.patch_sizes[1] * n_channels)

        loss = (x - target) ** 2
        loss = loss.mean(dim=-1) 

        loss = (loss * mask).sum() / mask.sum()
        return loss