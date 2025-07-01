import torch
import torch.nn as nn
from einops import rearrange
from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator
import timm

class Model(BaseModel):
    def __init__(self, vocabulary, embedding_dim=256, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__(vocabulary=vocabulary)
        self.embedding_dim = embedding_dim

        self.image_encoder = ImageEncoder(embedding_dim=self.embedding_dim)
        self.caption_generator = CaptionGenerator(
            vocabulary_size=len(self.vocabulary),
            embedding_dim=self.embedding_dim,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )


class ImageEncoder(BaseImageEncoder):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.resnet = timm.create_model("resnet50", pretrained=True, features_only=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Linear(2048, self.embedding_dim)

    def freeze(self):
        pass  # frozen by default

    def forward(self, image):
        features = self.resnet(image)[-1]
        pooled = self.avgpool(features).squeeze(-1).squeeze(-1)
        encoded = self.projector(pooled)
        return encoded


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, num_layers, nhead, dim_feedforward, dropout):
        super().__init__(vocabulary_size=vocabulary_size)

        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dim)

        # TransformerEncoderLayer as decoder:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # to keep (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_logits = nn.Linear(embedding_dim, vocabulary_size)

        # Map encoded image to a start token embedding or to condition transformer
        self.image_to_embedding = nn.Linear(embedding_dim, embedding_dim)

    def freeze(self):
        pass

    def forward(self, encoded_image, caption_indices, *args):
        """
        encoded_image: (B, embedding_dim)
        caption_indices: (B, seq_len)
        """
        batch_size, seq_len = caption_indices.shape

        # Embed captions tokens
        caption_embeddings = self.embedding(caption_indices)  # (B, seq_len, embedding_dim)

        # Condition the transformer on image encoding:
        # Option 1: prepend image embedding as a token to the sequence
        img_emb = self.image_to_embedding(encoded_image).unsqueeze(1)  # (B, 1, embedding_dim)
        transformer_input = torch.cat([img_emb, caption_embeddings], dim=1)  # (B, seq_len+1, embedding_dim)

        # Generate attention mask to prevent attending to future tokens
        # Mask shape should be (seq_len+1, seq_len+1)
        seq_length = seq_len + 1
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(encoded_image.device)

        transformer_output = self.transformer_encoder(transformer_input, mask=mask)  # (B, seq_len+1, embedding_dim)

        # We only decode logits for the tokens (skip the image embedding output)
        logits = self.to_logits(transformer_output[:, 1:, :])  # (B, seq_len, vocab_size)

        # Rearrange logits shape to (B, vocab_size, seq_len)
        logits = rearrange(logits, 'b s v -> b v s')

        indices = logits.argmax(dim=1)  # (B, seq_len)

        return {'logits': logits, 'indices': indices}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        device = encoded_image.device
        caption_indices = []

        # Initialize input with SOS token (batch size 1)
        input_tokens = torch.tensor([[sos_token_index]], device=device)  # (1, 1)

        for _ in range(max_length):
            output = self.forward(encoded_image=encoded_image,
                                  caption_indices=input_tokens)
            logits = output['logits']  # (1, vocab_size, seq_len)
            last_logits = logits[:, :, -1]  # (1, vocab_size)
            predicted_index = last_logits.argmax(dim=1)  # (1,)

            caption_indices.append(predicted_index.item())

            if predicted_index.item() == eos_token_index:
                break

            # Append predicted index to input tokens for next step
            input_tokens = torch.cat([input_tokens, predicted_index.unsqueeze(1)], dim=1)

        return caption_indices
