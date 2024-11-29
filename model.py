import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import numpy as np
import math
from math import ceil
import copy

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # Now x has size (batch_size,num_heads,seq_len,head_dim)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        return attention_output, weights

    
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3, max_patches=16):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.max_paches = max_patches

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
    
        # Patch embeddings
        x = self.patch_embeddings(x)  # Shape: [B, hidden_size, h_patches, w_patches]
        x = x.flatten(2).transpose(-1, -2)  # Shape: [B, n_patches, hidden_size]
    
        # Add class token
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [B, n_patches+1, hidden_size]
    
        # Interpolate positional embeddings dynamically
        num_patches = x.shape[1]  # Number of patches + 1 for class token
        pos_embed = self.interpolate_positional_embeddings(self.position_embeddings, num_patches)
    
        # Add positional embeddings
        x += pos_embed
        x = self.dropout(x)
        return x


    def interpolate_positional_embeddings(self, pos_embs, new_max_seq_len):
        # Reshape to add batch and channel dimensions required by interpolate
        pos_embs = pos_embs.transpose(1, 2)  # Shape: (1, hidden_size, max_seq_len)
        # Perform interpolation
        pos_embs = F.interpolate(pos_embs, size=new_max_seq_len, mode='linear', align_corners=False)
        # Reshape back to original dimensions
        pos_embs = pos_embs.transpose(1, 2)    # Shape: (1, new_max_seq_len, hidden_size)
        return pos_embs


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

    
class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights

    
class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=28, num_classes=10, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights


        
        
        
        
def pad_image(image, grid_size=(4, 4)):

    if len(image.size()) == 4:  # [batch_size, channels, height, width]
        batch_size, _, h, w = image.size()
    elif len(image.size()) == 3:  # [channels, height, width]
        h, w = image.size()[-2:]
    else:
        raise ValueError(f"Invalid image shape: {image.size()}")

    # calculate padding
    patch_h = ceil(h / grid_size[0])
    patch_w = ceil(w / grid_size[1])
    target_h = patch_h * grid_size[0]
    target_w = patch_w * grid_size[1]
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left


    padded_image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
    return padded_image


def split_into_patches(image, grid_size=(4, 4)):
    if len(image.size()) == 4:  # [batch_size, channels, height, width]
        batch_size, channels, h, w = image.size()
    elif len(image.size()) == 3:  # [channels, height, width]
        channels, h, w = image.size()
        batch_size = None
    else:
        raise ValueError(f"Invalid image shape: {image.size()}")

    patch_h = h // grid_size[0]
    patch_w = w // grid_size[1]

    patches = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            patch = image[..., i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]
            patches.append(patch.unsqueeze(0))  # 添加维度以支持批量堆叠

    if batch_size is not None:
        return torch.cat(patches, dim=0).reshape(batch_size, grid_size[0] * grid_size[1], channels, patch_h, patch_w)
    else:
        return torch.cat(patches, dim=0)


class PatchCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=12, kernel_size=4):
        super(PatchCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class SPP(nn.Module):
    def __init__(self, pool_sizes=(1, 2, 4)):
        super(SPP, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):

        if len(x.size()) == 3:
            x = x.unsqueeze(0)

        batch_size, channels, _, _ = x.size()
        features = []

        for pool_size in self.pool_sizes:
            pooled = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
            flattened = pooled.view(batch_size, -1)
            features.append(flattened)

        return torch.cat(features, dim=1)
    
class SPP_Embeddings(nn.Module):
    """Construct the embeddings from CNN + SPP for Vision Transformer."""
    def __init__(self, config, img_size, in_channels=3):
        super(SPP_Embeddings, self).__init__()

        self.grid_size = (4, 4)
        self.cnn = PatchCNN(in_channels=in_channels, out_channels=12, kernel_size=4)
        self.spp = SPP(pool_sizes=(1, 2, 4))

        self.hidden_size = config.hidden_size
        self.fc = nn.Linear(252, self.hidden_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, 17, self.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        self.dropout = nn.Dropout(config.transformer.dropout_rate)

    def forward(self, x):
        print(f"Input Shape: {x.shape}")

        # 1. Padding
        x = pad_image(x, grid_size=self.grid_size)
        print(f"Shape after Padding: {x.shape}")

        # 2. Split into patches
        patches = [split_into_patches(img, grid_size=self.grid_size) for img in x]
        print(f"Number of Patches: {len(patches)}")

        # 3. Process each patch with CNN+SPP
        patch_features = []
        for img_patches in patches:
            img_patch_features = []
            for patch in img_patches:
#                 print(f"Patch Shape Before CNN: {patch.shape}")
                cnn_output = self.cnn(patch)
#                 print(f"Patch Shape After CNN: {cnn_output.shape}")
                spp_output = self.spp(cnn_output)
#                 print(f"Patch Shape After SPP: {spp_output.shape}")
                img_patch_features.append(spp_output)
            patch_features.append(torch.stack(img_patch_features, dim=0))  # [num_patches, hidden_size]

        # 4. Stack into batch
        patch_features = torch.stack(patch_features, dim=0)  # [batch_size, num_patches, 1, hidden_size]
        patch_features = patch_features.squeeze(2)
        print(f"Shape after stacking patches: {patch_features.shape}")

        # 5. Map to Transformer hidden size
        patch_features = self.fc(patch_features)

        # 6. Add CLS Token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # [batch_size, 1, hidden_size]
        embeddings = torch.cat((cls_tokens, patch_features), dim=1)  # [batch_size, num_patches+1, hidden_size]

        # 7. Add position embeddings
        embeddings += self.position_embeddings
        embeddings = self.dropout(embeddings)

        print(f"Final Embedding Shape: {embeddings.shape}")
        return embeddings

class TransformerSPP(nn.Module):
    def __init__(self, config, img_size, vis):
        super(TransformerSPP, self).__init__()
        self.embeddings = SPP_Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights
    
    
class VisionTransformerSPP(nn.Module):
    def __init__(self, config, img_size=28, num_classes=10, zero_head=False, vis=False):
        super(VisionTransformerSPP, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = TransformerSPP(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights