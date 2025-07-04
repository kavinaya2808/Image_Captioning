import torch
import torch.nn
import timm
from einops import rearrange, pack
from models.base import BaseModel, BaseImageEncoder, BaseCaptionGenerator


class Model(BaseModel):
    def __init__(self, vocabulary, embedding_dim, num_layers):
        super().__init__(vocabulary=vocabulary)

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.image_encoder = ImageEncoder(embedding_dim=self.embedding_dim)
        self.caption_generator = CaptionGenerator(vocabulary_size=len(self.vocabulary),
                                                  embedding_dim=self.embedding_dim,
                                                  hidden_dim=self.embedding_dim,
                                                  num_layers=self.num_layers)


class ImageEncoder(BaseImageEncoder):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False

        self.projector = torch.nn.Linear(self.vit.embed_dim, self.embedding_dim)

    def freeze(self):
        pass  # already frozen

    def forward(self, image):
        with torch.no_grad():
            vit_output = self.vit.forward_features(image)  # (B, D)
        cls_token = vit_output[:, 0, :]  # extract [CLS] token
        return self.projector(cls_token)  # (B, embedding_dim)


class CaptionGenerator(BaseCaptionGenerator):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_layers):
        super().__init__(vocabulary_size=vocabulary_size)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_dim),
            torch.nn.Dropout(0.5)
        )

        # 🔄 Replacing RNN with LSTM
        self.rnn = torch.nn.LSTM(input_size=self.embedding_dim,
                                 hidden_size=self.hidden_dim,
                                 num_layers=self.num_layers,
                                 batch_first=True)

        self.to_logits = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.vocabulary_size)

    def freeze(self):
        pass

    def _get_embeddings(self, encoded_image=None, caption_indices=None):
        if caption_indices is None:
            embeddings = rearrange(encoded_image, 'batch embedding_dim -> batch 1 embedding_dim')
        else:
            embeddings = self.embedding(caption_indices)
            if encoded_image is not None:
                embeddings, _ = pack([encoded_image, embeddings], 'batch * embedding_dim')

        return embeddings

    def forward(self, encoded_image, caption_indices, hidden_state=None):
        if encoded_image is not None and caption_indices is not None:
            caption_indices = caption_indices[:, 1:]  # the encoded image will be used instead of the <SOS> token

        embeddings = self._get_embeddings(encoded_image=encoded_image, caption_indices=caption_indices)

        # Support LSTM hidden state as a tuple
        if isinstance(self.rnn, torch.nn.LSTM):
            output, (h, c) = self.rnn(input=embeddings, hx=hidden_state if hidden_state else None)
            hidden_state = (h, c)
        else:
            output, hidden_state = self.rnn(input=embeddings, hx=hidden_state if hidden_state else None)

        logits = self.to_logits(output)
        logits = rearrange(logits, 'batch sequence_length vocabulary_size -> batch vocabulary_size sequence_length')

        return {'logits': logits, 'indices': logits.argmax(dim=-2), 'hidden_state': hidden_state}

    def generate_caption_indices(self, encoded_image, sos_token_index, eos_token_index, max_length):
        caption_indices = []

        output = self.forward(encoded_image, caption_indices=None, hidden_state=None)
        for _ in range(max_length):
            predicted_index = output['indices']
            caption_indices.append(predicted_index.item())

            if predicted_index.item() == eos_token_index:
                break

            output = self.forward(encoded_image=None,
                                  caption_indices=predicted_index,
                                  hidden_state=output['hidden_state'])

        return caption_indices
