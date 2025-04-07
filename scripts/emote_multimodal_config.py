import os
from emote_config import Emote_Config

class Emote_Multimodal_Config(Emote_Config):
    def __init__(self, finetune, model_name, portion, seed, hfpath, use_images=False, image_model_name="swin_tiny_patch4_window7_224"):
        super().__init__(finetune, model_name, portion, seed, hfpath)
        
        # Add image-specific configurations
        self.use_images = use_images
        self.image_model_name = image_model_name
        
        # Image directories
        self.train_img_dir = 'dataset_following_elco_split/generated_img_dataset/train_google'
        self.val_img_dir = 'dataset_following_elco_split/generated_img_dataset/val_google'
        self.test_img_dir = 'dataset_following_elco_split/generated_img_dataset/test_google'
        
        # Fusion parameters
        self.fusion_hidden_size = 768  # Same as BERT hidden size for compatibility
        self.fusion_dropout = 0.1
        
        # Model save directory for multimodal model
        self.model_save_dir = '{}/emote-multimodal-{}'.format(self.hfpath, self.model_name)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir) 