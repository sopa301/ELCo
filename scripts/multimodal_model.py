import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification
from torchvision.models import swin_transformer

class MultimodalFusion(nn.Module):
    def __init__(self, text_dim, img_dim, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Project image features to match text dimension
        self.image_projection = nn.Linear(img_dim, hidden_size)
        
        # Bidirectional cross-attention components
        # For text attending to image
        self.text_query = nn.Linear(hidden_size, hidden_size)
        self.img_key = nn.Linear(hidden_size, hidden_size)
        self.img_value = nn.Linear(hidden_size, hidden_size)
        
        # For image attending to text
        self.img_query = nn.Linear(hidden_size, hidden_size)
        self.text_key = nn.Linear(hidden_size, hidden_size)
        self.text_value = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization for better training stability
        self.text_norm = nn.LayerNorm(hidden_size)
        self.img_norm = nn.LayerNorm(hidden_size)
        
        # Final fusion layer with gating mechanism
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # Gating mechanism to control information flow
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.text_query, self.img_key, self.img_value, 
                      self.img_query, self.text_key, self.text_value, 
                      self.image_projection]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
    
    def forward(self, text_features, image_features):
        batch_size = text_features.shape[0]
        
        # Get CLS token from text representation
        cls_token = text_features[:, 0]  # [batch, hidden]
        
        # Project image features
        img_projected = self.image_projection(image_features)  # [batch, hidden]
        
        # 1. Image attending to text (img_query attends to text_key/value)
        img_q = self.img_query(img_projected).view(batch_size, 1, -1)
        text_k = self.text_key(text_features)
        text_v = self.text_value(text_features)
        
        img_attention_scores = torch.matmul(img_q, text_k.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        img_attention_weights = torch.softmax(img_attention_scores, dim=-1)
        img_context = torch.matmul(img_attention_weights, text_v).squeeze(1)  # [batch, hidden]
        img_context = self.img_norm(img_context + img_projected)  # Residual connection
        
        # 2. Text attending to image (text_query attends to img_key/value)
        # Use CLS token as query
        text_q = self.text_query(cls_token).view(batch_size, 1, -1)
        img_k = self.img_key(img_projected).view(batch_size, 1, -1)
        img_v = self.img_value(img_projected).view(batch_size, 1, -1)
        
        text_attention_scores = torch.matmul(text_q, img_k.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        text_attention_weights = torch.softmax(text_attention_scores, dim=-1)
        text_context = torch.matmul(text_attention_weights, img_v).squeeze(1)  # [batch, hidden]
        text_context = self.text_norm(text_context + cls_token)  # Residual connection
        
        # Combine both contexts with adaptive gating
        combined = torch.cat([text_context, img_context], dim=1)
        
        # Calculate modality weights through gating
        gate_weights = self.gate(combined)
        
        # Apply gating to control information flow from each modality
        text_weight = gate_weights[:, 0].unsqueeze(1)
        img_weight = gate_weights[:, 1].unsqueeze(1)
        
        weighted_text = text_context * text_weight
        weighted_img = img_context * img_weight
        
        # Concatenate weighted features
        multimodal_features = torch.cat([weighted_text, weighted_img], dim=1)
        
        # Final fusion
        output = self.fusion(multimodal_features)
        
        return output

class EmoteMultimodalModel(nn.Module):
    def __init__(self, config, num_labels=2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        # Text encoder
        self.text_model = AutoModel.from_pretrained(config.model_path)
        
        # Image encoder (Swin Transformer)
        if config.image_model_name == "swin_tiny_patch4_window7_224":
            self.image_model = swin_transformer.swin_t(weights="DEFAULT")
        elif config.image_model_name == "swin_small_patch4_window7_224":
            self.image_model = swin_transformer.swin_s(weights="DEFAULT")
        elif config.image_model_name == "swin_base_patch4_window7_224":
            self.image_model = swin_transformer.swin_b(weights="DEFAULT")
        else:
            raise ValueError(f"Unsupported image model: {config.image_model_name}")
        
        # Remove the classification head from the image model
        self.image_model.head = nn.Identity()
        
        # IMPORTANT: Freeze the image model completely
        for param in self.image_model.parameters():
            param.requires_grad = False
        
        # Get dimensions
        self.text_dim = 768  # BERT hidden size
        self.img_dim = self.image_model.norm.normalized_shape[0]
        
        # Multimodal fusion
        self.fusion = MultimodalFusion(self.text_dim, self.img_dim, config.fusion_hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.fusion_dropout)
        
        # Classification head
        self.classifier = nn.Linear(config.fusion_hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
        
        # Training stabilization attributes
        self.gradient_clip_val = 1.0
        self.lr_warmup_steps = 100
        self.total_steps = 0
    
    def _init_weights(self):
        """Initialize the weights properly"""
        # Initialize classifier
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, images=None, labels=None, **kwargs):
        # Process text input
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Get text features
        text_features = text_outputs.last_hidden_state
        
        # Process image input
        image_features = self.image_model(images)
        
        # Fuse modalities
        fused_features = self.fusion(text_features, image_features)
        
        # Apply dropout
        fused_features = self.dropout(fused_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Calculate loss
        loss = None
        if labels is not None:
            # Use CrossEntropyLoss 
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
            # For stability: if loss is too high, reduce it but keep gradients proportional
            if loss.item() > 10:
                loss = loss * (10 / loss.item())
        
        # Increment step counter for learning rate scheduling
        self.total_steps += 1
        
        # Return in a format compatible with the existing code
        return type('obj', (object,), {
            'loss': loss,
            'logits': logits,
            'hidden_states': text_outputs.hidden_states
        })
    
    # Methods for saving and loading
    def save_pretrained(self, save_directory):
        """Save the model to the specified directory."""
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the model configuration
        model_config = {
            "image_model_name": self.config.image_model_name,
            "num_labels": self.num_labels
        }
        
        # Save text model
        self.text_model.save_pretrained(os.path.join(save_directory, "text_model"))
        
        # Save other components
        torch.save(
            {
                "model_config": model_config,
                "image_model_state_dict": self.image_model.state_dict(),
                "classifier_state_dict": self.classifier.state_dict(),
            },
            os.path.join(save_directory, "multimodal_components.pt")
        )
        
        return save_directory
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, config, **kwargs):
        """Load a model from the specified directory."""
        # Load model configuration and components
        components_path = os.path.join(pretrained_model_path, "multimodal_components.pt")
        if not os.path.exists(components_path):
            raise ValueError(f"Could not find multimodal components at {components_path}")
        
        components = torch.load(components_path, map_location="cpu")
        model_config = components["model_config"]
        
        # Create new instance
        num_labels = model_config.get("num_labels", 2)
        model = cls(config, num_labels=num_labels)
        
        # Load text model
        model.text_model = AutoModel.from_pretrained(os.path.join(pretrained_model_path, "text_model"))
        
        # Load other components
        model.image_model.load_state_dict(components["image_model_state_dict"])
        model.classifier.load_state_dict(components["classifier_state_dict"])
        
        return model 