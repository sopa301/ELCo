import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class MultimodalDataset(Dataset):
    def __init__(self, encodings, labels, strategies, img_dir, img_transform=None, debug=False):
        self.encodings = encodings
        self.labels = labels
        self.strategies = strategies
        self.img_dir = img_dir
        self.debug = False 
        
        # Load tokenizer once if debug is enabled
        if self.debug and any('input_ids' in enc for enc in [encodings]):
            from transformers import AutoTokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
        
        # Default image transformation if none provided
        if img_transform is None:
            self.img_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = img_transform
            
        # Create a placeholder image (black image)
        self.placeholder_image = torch.zeros(3, 224, 224)
        
        # Apply normalization to the placeholder image to match other images
        norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.placeholder_image = (self.placeholder_image - norm_mean) / norm_std
        
        # Filter out indices where images don't exist
        self.valid_indices = []
        for idx in range(len(self.labels)):
            img_path = os.path.join(self.img_dir, f"{idx}.png")
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
        
        if len(self.valid_indices) < len(self.labels):
            print(f"Filtered out {len(self.labels) - len(self.valid_indices)} datapoints without images.")
            print(f"Dataset size reduced from {len(self.labels)} to {len(self.valid_indices)}.")

    def __getitem__(self, idx):
        # Map the requested index to a valid index
        original_idx = self.valid_indices[idx]
        
        item = {key: val[original_idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[original_idx].clone().detach()
        item['strategies'] = self.strategies[original_idx].clone().detach()
        
        # Load image
        img_path = os.path.join(self.img_dir, f"{original_idx}.png")
        
        # Debug info
        if self.debug:
            # Extract text info if possible (from input_ids)
            if 'input_ids' in item and self.tokenizer is not None:
                try:
                    # Convert to CPU and integer if needed
                    input_ids = item['input_ids'].cpu().int().tolist()
                    text = self.tokenizer.decode(input_ids)
                    print(f"Index: {original_idx}, Image: {img_path}, Label: {item['labels'].item()}, Strategy: {item['strategies'].item()}")
                    print(f"Text: {text}")
                    print("-" * 50)
                except Exception as e:
                    print(f"Index: {original_idx}, Image: {img_path}, Label: {item['labels'].item()}, Strategy: {item['strategies'].item()}")
                    print(f"Could not decode text: {str(e)}")
                    print("-" * 50)
            else:
                print(f"Index: {original_idx}, Image: {img_path}, Label: {item['labels'].item()}, Strategy: {item['strategies'].item()}")
                print("-" * 50)
        
        # The image should always exist since we filtered the indices
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.img_transform(image)
        except Exception as e:
            print(f"Warning: Could not process image {img_path} despite it existing. Error: {e}")
            image = self.placeholder_image
        
        item['images'] = image
        return item

    def __len__(self):
        return len(self.valid_indices) 