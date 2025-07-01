import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

# Attention Visualization
def visualize_attention(image_path, caption, attention_weights, token_indices, vocabulary, save_path=None):
    img = Image.open(image_path).convert("RGB")
    img_size = img.size

    num_plots = min(3, len(token_indices))
    fig, axes = plt.subplots(1, num_plots + 1, figsize=(5 * (num_plots + 1), 5))
    fig.suptitle(f"Generated Caption: {caption}", y=1.05)

    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    for i, token_idx in enumerate(token_indices[:num_plots]):
        if token_idx >= len(attention_weights):
            continue

        attn = attention_weights[token_idx]  # Shape: [num_patches]
        attn_np = attn.detach().cpu().numpy()

        length = attn_np.shape[0]
        side = int(np.sqrt(length))
        if side * side != length:
            print(f"Skipping token {token_idx}: attention shape {attn_np.shape} not square")
            continue

        attn_map = attn_np.reshape(side, side)
        attn_img = Image.fromarray(attn_map).resize(img_size, Image.BILINEAR)
        attn_img = np.array(attn_img)
        attn_img = (attn_img - attn_img.min()) / (attn_img.max() - attn_img.min() + 1e-8)

        axes[i + 1].imshow(img)
        axes[i + 1].imshow(attn_img, cmap='jet', alpha=0.5)
        token_text = vocabulary.to_token(token_indices[i])
        axes[i + 1].set_title(f"Attention: '{token_text}'")
        axes[i + 1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

# Generate caption and capture attention
def generate_and_visualize(model, image_path, vocabulary, sos_token_idx, eos_token_idx, max_length=20, save_path=None):
    model = model.to(device)
    model.eval()
    image_tensor = preprocess_image(image_path).to(device)

    decoder_layers = model.caption_generator.transformer_decoder.layers
    attention_maps = []

    # Monkey-patch MultiheadAttention.forward to capture attn weights
    original_forward = nn.MultiheadAttention.forward

    def custom_forward(self, query, key, value, **kwargs):
        kwargs["need_weights"] = True  # <=== this line ensures attention is returned
        output, attn_weights = original_forward(self, query, key, value, **kwargs)
        self._last_attn_weights = attn_weights
        return output, attn_weights


    # Patch the final decoder layer only (we’ll read attention from this one)
    last_layer = decoder_layers[-1]
    last_layer.multihead_attn.forward = custom_forward.__get__(last_layer.multihead_attn, nn.MultiheadAttention)

    with torch.no_grad():
        encoded_image = model.image_encoder(image_tensor)  # [1, num_patches, 768]
        memory = model.caption_generator.image_linear(encoded_image).permute(1, 0, 2)  # [num_patches, 1, D]

        generated_indices = [sos_token_idx]

        for _ in range(max_length):
            current_input = torch.tensor(generated_indices).unsqueeze(0).to(device)  # [1, seq_len]
            tgt = model.caption_generator.embedding(current_input)
            tgt = model.caption_generator.positional_encoding(tgt)
            tgt = tgt.permute(1, 0, 2)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(device)

            output = model.caption_generator.transformer_decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask
            )

            logits = model.caption_generator.output_layer(output.permute(1, 0, 2))  # [1, seq_len, vocab]
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated_indices.append(next_token)

            # Capture cross-attention for this step (from last decoder layer)
            if hasattr(last_layer.multihead_attn, "_last_attn_weights"):
                # Shape: [1, seq_len, num_patches] → we take the last token's weights
                attn = last_layer.multihead_attn._last_attn_weights[0, -1, :]  # [num_patches]
                attention_maps.append(attn)

            if next_token == eos_token_idx:
                break

    caption = ' '.join([vocabulary.to_token(idx) for idx in generated_indices])

    # Skip punctuations, SOS
    valid_indices = [i for i, idx in enumerate(generated_indices)
                     if vocabulary.to_token(idx) not in ["<SOS>", "<EOS>", ".", ",", "!", "?"] and i > 0]
    selected_indices = valid_indices[:3]

    visualize_attention(
        image_path=image_path,
        caption=caption,
        attention_weights=attention_maps,
        token_indices=selected_indices,
        vocabulary=vocabulary,
        save_path=save_path
    )

    # Restore original method (optional)
    last_layer.multihead_attn.forward = original_forward.__get__(last_layer.multihead_attn, nn.MultiheadAttention)

    return caption
