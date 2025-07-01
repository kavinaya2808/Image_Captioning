import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator
import torch.nn.functional as F
import timm

class Model(BaseModel):
    """Model 6 using DINOv2 spatial tokens and Cross-Attention."""
    def __init__(self, vocabulary, embedding_dim=256, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__(vocabulary=vocabulary)  # MUST be called first

        self.image_encoder = ImageEncoder()
        self.caption_generator = CaptionGenerator(
            vocabulary_size=len(vocabulary),
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

    def freeze(self):
        # Freeze both image encoder and caption generator if needed
        self.image_encoder.freeze()
        # If you want to freeze caption generator embeddings, add here:
        # self.caption_generator.freeze()  # optional


class ImageEncoder(BaseImageEncoder):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.backbone.eval()

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, image):
        with torch.no_grad():
            x = self.backbone.forward_features(image)  # shape: (B, 1+num_patches, 768)
            spatial_tokens = x[:, 1:, :]  # remove CLS token
        return spatial_tokens

class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, num_layers, nhead, dim_feedforward, dropout):
        super().__init__(vocabulary_size=vocabulary_size)  # Make sure to call parent's constructor

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embedding_dim, vocabulary_size)
        self.image_linear = nn.Linear(768, embedding_dim)

    def forward(self, encoded_image, caption_indices, *args):
        tgt = self.embedding(caption_indices)  # [B, seq_len, D]
        tgt = self.positional_encoding(tgt)

        memory = self.image_linear(encoded_image)  # [B, num_patches, D]
        memory = memory.permute(1, 0, 2)           # [num_patches, B, D]
        tgt = tgt.permute(1, 0, 2)                 # [seq_len, B, D]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

        output = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        output = output.permute(1, 0, 2)  # [B, seq_len, D]

        logits = self.output_layer(output)  # [B, seq_len, vocab_size]
        indices = torch.argmax(logits, dim=-1)

        return {
            'logits': logits.permute(0, 2, 1),  # [B, vocab_size, seq_len]
            'indices': indices
        }

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        generated_indices = [sos_token_index]
        memory = self.image_linear(encoded_image)  # [1, num_patches, D]
        memory = memory.permute(1, 0, 2)

        for _ in range(max_length):
            current_input = torch.tensor(generated_indices).unsqueeze(0).to(encoded_image.device)
            tgt = self.embedding(current_input)
            tgt = self.positional_encoding(tgt)
            tgt = tgt.permute(1, 0, 2)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

            output = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            output = output.permute(1, 0, 2)
            logits = self.output_layer(output)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated_indices.append(next_token)

            if next_token == eos_token_index:
                break

        return generated_indices


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
