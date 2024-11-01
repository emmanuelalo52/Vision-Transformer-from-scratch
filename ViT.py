import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from datasets import load_dataset
from transformers import ViTFeatureExtractor
@dataclass
class VITConfig:
    n_emb: int = 768
    image_size: int = 224
    n_heads: int = 12
    patch_size: int = 16
    n_layers: int = 12
    num_patches: int = (image_size // patch_size) ** 2
    num_classes: int = 10


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Device: {device}")




class SelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_emb % config.n_heads == 0
        self.n_head = config.n_heads
        self.n_emb = config.n_emb
        self.n_emb = config.n_emb
        self.proj = nn.Linear(config.n_emb,config.n_emb)
        self.scale = config.n_emb ** -0.5
        self.c_attn = nn.Linear(config.n_emb, 3 * config.n_emb)
        
    def forward(self,x):
        B, N, C = x.size() # batch size, sequence length, embedding dimensionality (n_emb   )
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
    
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_emb, 4 * config.n_emb)
        self.gelu = nn.GELU(approximate='tanh') #for our large dataset regularization
        self.fc2 = nn.Linear(4 * config.n_emb, config.n_emb)
        self.dropout = nn.Dropout()
    def forward(self,x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(config)
        self.msa = SelfAttention(config)
        self.norm1 = nn.LayerNorm(config.n_emb)
        self.norm2 = nn.LayerNorm(config.n_emb)

    def forward(self, x):
        x = x + self.msa(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    


# Input image is (B, C, H, W)
# And patch_size is (P, P)
# B = batch size, C = channels, H = height, W = width, P = patch size
class PatchEmbedding(nn.Module):
    def __init__(self,config):
        super().__init__()
        #in_channels(n_hidden = n_emb/n_heads)
        self.ln_proj = nn.Conv2d(
            in_channels=3,
            out_channels=config.n_emb,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        self.flatten = nn.Flatten(2)
    def forward(self,x):
        # x size = (B, C, H, W)
        x = self.ln_proj(x) # (B, embed_dim, H//patch_size, W//patch_size)
        x = self.flatten(x) # (B, embed_dim, H*W//patch_size^2)
        x = x.transpose(1,2) # (B, H*W//patch_size^2, embed_dim)
        return x

# vision transformer
#includes the ecoder and cls token, position embedding.

class Positional2DEmbedding(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.h, self.w = int(torch.sqrt(torch.tensor(config.num_patches))), int(torch.sqrt(torch.tensor(config.num_patches)))
        self.x_emb = nn.Parameter(torch.zeros(1,self.h,config.n_emb//2))
        self.y_emb = nn.Parameter(torch.zeros(self.w,1,config.n_emb//2))

        #cls token
        self.cls_token = nn.Parameter(torch.zeros(1,1,config.n_emb))

        #initalize our values
        nn.init.trunc_normal_(self.x_emb, std=0.02)
        nn.init.trunc_normal_(self.y_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self,x):
        B,N,C = x.shape
        #We want to set up our parameters for broadcasting across our dataset
        x_pos = self.x_emb.expand(self.h,-1,-1)
        y_pos = self.y_emb.expand(-1,self.w,-1)
        #concatenate
        pos_emb = torch.concatenate([x_pos,y_pos],dim=-1)
        pos_emb = pos_emb.reshape(-1,C)
        x = x + pos_emb.unsqueeze(0)
        cls_token_pos_emb = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token_pos_emb, x], dim=1)
        return x
    


class ViT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformerencoder = nn.ModuleDict(dict(wpe = PatchEmbedding(config),
                                                     wps = Positional2DEmbedding(config),
                                                     hidden_layer = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                                                     ln = nn.LayerNorm(config.n_emb),))
        self.n_heads = nn.Linear(config.n_emb,config.patch_size,bias=False)
        #share weight of output embedding at the beginning of the layer and at the pre-softmax stage
        self.transformerencoder.wps.weight = self.n_heads.weight
        #initalise parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'weight'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.transformerencoder['wpe'](x)
        x = self.transformerencoder['wps'](x)
        for block in self.transformerencoder['hidden_layer']:
            x = block(x)
        x = self.transformerencoder['ln'](x)
        cls_output = x[:, 0]  # Take the class token output
        x = self.n_heads(cls_output)
        return x

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Example usage
config = VITConfig()
model = ViT(config).to(device)
print(model)

# Compile the model
model = torch.compile(model)

# Data loading and preprocessing
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

def preprocess(example):
    image = example['img']  # Corrected key to 'img'
    image = feature_extractor(images=image, return_tensors="pt")['pixel_values'].squeeze()
    return {'image': image, 'label': example['label']}

# Load a small dataset from Hugging Face
dataset = load_dataset('cifar10')
train_dataset = dataset['train'].select(range(1000)).map(preprocess, batched=True)
test_dataset = dataset['test'].select(range(1000)).map(preprocess, batched=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Loss function and optimizer
# Setup AdamW optimizer and a learning rate scheduler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_step(model, dataloader, loss_fn, optimizer, scheduler, clip_grad=1.0):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch['images'].to(device))
        loss = loss_fn(outputs, batch['labels'].to(device))
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        scheduler.step()
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)

# Inference loop
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = running_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    return test_loss, test_accuracy

# Training and evaluation
num_epochs = 5
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')