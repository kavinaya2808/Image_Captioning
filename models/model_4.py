import torch
import torch.nn
from einops import rearrange
from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator
import timm

class Model(BaseModel):
    def __init__(self, vocabulary, embedding_dim, hidden_dim, num_layers):
        super().__init__(vocabulary=vocabulary)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.image_encoder = ImageEncoder(embedding_dim=self.embedding_dim)
        self.caption_generator = CaptionGenerator(vocabulary_size=len(self.vocabulary),
                                                  embedding_dim=self.embedding_dim,
                                                  hidden_dim=self.hidden_dim,
                                                  num_layers=self.num_layers)

class ImageEncoder(BaseImageEncoder):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Use a ViT model pretrained, e.g., vit_base_patch16_224 (like Model 3)
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False

        self.projector = torch.nn.Linear(self.vit.embed_dim, self.embedding_dim)

    def freeze(self):
        pass  # frozen by default

    def forward(self, image):
        with torch.no_grad():
            vit_output = self.vit.forward_features(image)  # (B, D)
        cls_token = vit_output[:, 0, :]  # CLS token
        return self.projector(cls_token)  # (B, embedding_dim)


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers):
        super().__init__(vocabulary_size=vocabulary_size)

        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dim),
            torch.nn.Dropout(0.5)
        )

        self.rnn = torch.nn.LSTM(input_size=embedding_dim,
                                 hidden_size=hidden_dim,
                                 num_layers=num_layers,
                                 batch_first=True)

        self.to_logits = torch.nn.Linear(hidden_dim, vocabulary_size)

        # NEW: linear layers to map encoded image to initial hidden and cell states
        self.init_hidden = torch.nn.Linear(embedding_dim, hidden_dim * num_layers)
        self.init_cell = torch.nn.Linear(embedding_dim, hidden_dim * num_layers)

    def freeze(self):
        pass

    def _init_hidden_states(self, encoded_image):
        """Convert encoded image to initial LSTM hidden and cell states."""
        batch_size = encoded_image.size(0)

        h = self.init_hidden(encoded_image)  # (B, hidden_dim * num_layers)
        c = self.init_cell(encoded_image)

        # reshape to (num_layers, batch, hidden_dim)
        h = h.view(self.num_layers, batch_size, self.hidden_dim)
        c = c.view(self.num_layers, batch_size, self.hidden_dim)

        return (h, c)

    def forward(self, encoded_image, caption_indices, hidden_state=None):
        # We no longer prepend encoded_image as first token.
        # Instead, initialize hidden state from encoded_image.
        embeddings = self.embedding(caption_indices)  # (B, seq_len, embedding_dim)

        if hidden_state is None:
            hidden_state = self._init_hidden_states(encoded_image)  # init hidden state from image

        output, hidden_state = self.rnn(embeddings, hidden_state)  # output: (B, seq_len, hidden_dim)

        logits = self.to_logits(output)  # (B, seq_len, vocab_size)
        logits = rearrange(logits, 'batch seq_len vocab -> batch vocab seq_len')

        indices = logits.argmax(dim=1)  # (B, seq_len)

        return {'logits': logits, 'indices': indices, 'hidden_state': hidden_state}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        caption_indices = []

        # Start with <SOS> token
        input_token = torch.tensor([[sos_token_index]], device=encoded_image.device)

        hidden_state = self._init_hidden_states(encoded_image)

        for _ in range(max_length):
            output = self.forward(encoded_image=None,
                                  caption_indices=input_token,
                                  hidden_state=hidden_state)

            predicted_index = output['indices'][:, -1]  # last predicted token index
            caption_indices.append(predicted_index.item())

            if predicted_index.item() == eos_token_index:
                break

            input_token = predicted_index.unsqueeze(1)  # (B, 1)
            hidden_state = output['hidden_state']

        return caption_indices
