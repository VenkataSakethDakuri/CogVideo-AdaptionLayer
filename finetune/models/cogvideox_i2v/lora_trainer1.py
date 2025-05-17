from typing import Any, Dict, List, Tuple

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from numpy import dtype
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model

from ..utils import register


class CogVideoXI2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXImageToVideoPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )

        components.transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer"
        )

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        
        # Add emotion enhancement module
        num_layers = len(components.transformer.transformer_blocks)
        hidden_size = components.transformer.transformer_blocks[0].attn1.to_q.in_features
        components.emotion_enhancer = EmotionEnhancementModule(
            hidden_size=hidden_size,
            num_layers=num_layers,
            adaptation_factor=0.3,  # Can be configured via arguments
        )

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline:
        pipe = CogVideoXImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(
            prompt_token_ids.to(self.accelerator.device)
        )[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "images": []}

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["images"].append(image)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["images"] = torch.stack(ret["images"])

        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        images = batch["images"]

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        # Shape of images: [B, C, H, W]

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = images.unsqueeze(2)
        # Add noise to images
        image_noise_sigma = torch.normal(
            mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device
        )
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = (
            images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        )
        image_latent_dist = self.components.vae.encode(
            noisy_images.to(dtype=self.components.vae.dtype)
        ).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        )
        timesteps = timesteps.long()

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        assert (latent.shape[0], *latent.shape[2:]) == (
            image_latents.shape[0],
            *image_latents.shape[2:],
        )

        # Padding image_latents to the same frame number as latent
        padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Add noise to latent
        noise = torch.randn_like(latent)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Concatenate latent and image_latents in the channel dimension
        latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None
            if self.state.transformer_config.ofs_embed_dim is None
            else latent.new_full((1,), fill_value=2.0)
        )
        # Get the current prompt for emotion enhancement (using first sample in batch)
        if hasattr(self.state, 'current_prompts') and len(self.state.current_prompts) > 0:
            self.current_prompt = self.state.current_prompts[0]
        else:
            self.current_prompt = ""  # Default if no prompt available

        # Apply transformer with emotion enhancement
        predicted_noise = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        # Denoise
        latent_pred = self.components.scheduler.get_velocity(
            predicted_noise, latent_noisy, timesteps
        )

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        return loss

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXImageToVideoPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]
        
        # Set current prompt for emotion enhancement
        self.current_prompt = prompt
        
        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=prompt,
            image=image,
            generator=self.state.generator,
        ).frames[0]
        return [("video", video_generate)]

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (
                num_frames + transformer_config.patch_size_t - 1
            ) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin

    def setup_emotion_enhancement_hooks(self):
        """Set up hooks to apply emotion enhancement during inference"""
        # Store original forward methods
        if not hasattr(self, 'original_forward_methods'):
            self.original_forward_methods = {}
        
        transformer = unwrap_model(self.accelerator, self.components.transformer)
        
        # For each transformer block that we want to enhance
        for layer_idx, block in enumerate(transformer.transformer_blocks):
            if layer_idx % 3 == 0:  # Apply to every third layer
                # Save original forward method
                if block not in self.original_forward_methods:
                    self.original_forward_methods[block] = block.forward
                    
                    # Define new forward method with emotion enhancement
                    def make_forward_with_emotion(original_forward, layer_idx):
                        def forward_with_emotion(hidden_states, *args, **kwargs):
                            # Apply original forward pass
                            hidden_states = original_forward(hidden_states, *args, **kwargs)
                            
                            # Get current prompt
                            prompt = getattr(self, 'current_prompt', '')
                            
                            # Apply emotion enhancement
                            if hasattr(self.components, 'emotion_enhancer'):
                                hidden_states = self.components.emotion_enhancer.enhance_emotions(
                                    hidden_states, prompt, layer_idx
                                )
                            
                            return hidden_states
                        
                        return forward_with_emotion
                    
                    # Replace forward method
                    block.forward = make_forward_with_emotion(
                        self.original_forward_methods[block], layer_idx
                    )

    @override
    def on_fit_start(self):
        super().on_fit_start()
        # Set up emotion enhancement hooks
        self.setup_emotion_enhancement_hooks()
        
        # Store prompt from each batch for emotion enhancement
        self.state.current_prompts = []

    @override
    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()
        # Wrap dataloader to capture prompts
        original_collate = dataloader.collate_fn
        
        def collate_with_prompts(batch):
            # Extract prompts if available
            if batch and isinstance(batch[0], dict) and 'prompt' in batch[0]:
                self.state.current_prompts = [item['prompt'] for item in batch]
            return original_collate(batch)
        
        dataloader.collate_fn = collate_with_prompts
        return dataloader


class EmotionEnhancementModule(nn.Module):
    """
    Module that enhances emotional expressions in human subjects within videos.
    Focuses on facial expressions and body language.
    """

    def __init__(
        self,
        hidden_size,
        num_layers,
        adaptation_factor=0.3,
        emotion_vocab_size=24,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.adaptation_factor = adaptation_factor

        # Common emotions that might appear in prompts
        self.emotions = [
            "happy",
            "sad",
            "angry",
            "surprised",
            "scared",
            "nervous",
            "excited",
            "tired",
            "calm",
            "disgusted",
            "bored",
            "confident",
            "anxious",
            "relaxed",
            "confused",
            "serious",
            "smiling",
            "laughing",
            "crying",
            "terrified",
            "joyful",
            "worried",
            "proud",
            "neutral",
        ]

        # Emotion embeddings
        self.emotion_embeddings = nn.Embedding(emotion_vocab_size, hidden_size)

        # Human feature detection layers (face, body posture)
        self.human_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )

        # Emotion enhancement modules per layer
        self.emotion_modules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size // 4),
                    nn.SiLU(),
                    nn.Linear(hidden_size // 4, hidden_size),
                )
                for _ in range(num_layers // 3)  # Apply to every third layer to save parameters
            ]
        )

        # Attention for focusing on facial regions
        self.face_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights of the module"""
        nn.init.normal_(self.emotion_embeddings.weight, std=0.02)

        for module in self.human_detector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        for module_list in self.emotion_modules:
            for module in module_list:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    def extract_emotions(self, prompt):
        """Extract emotions from text prompt"""
        found_emotions = []
        for emotion in self.emotions:
            if emotion in prompt.lower():
                found_emotions.append(emotion)

        # Default to neutral if no emotions found
        if not found_emotions:
            found_emotions = ["neutral"]

        return found_emotions

    def enhance_emotions(self, hidden_states, prompt, layer_idx):
        """Apply emotion enhancement to hidden states"""
        # Extract emotions from prompt
        emotions = self.extract_emotions(prompt)

        if not emotions:
            return hidden_states

        # Get module index (apply to every third layer)
        module_idx = layer_idx // 3
        if module_idx >= len(self.emotion_modules):
            return hidden_states

        # Get emotion embeddings and average them
        emotion_indices = [self.emotions.index(e) for e in emotions if e in self.emotions]
        if not emotion_indices:
            return hidden_states

        emotion_idx_tensor = torch.tensor(
            emotion_indices, device=hidden_states.device
        )
        emotion_embedding = self.emotion_embeddings(emotion_idx_tensor).mean(dim=0)

        # Apply human detection - create attention query
        human_query = self.human_detector(emotion_embedding).unsqueeze(0).unsqueeze(0)

        # Use attention to focus on human/facial regions
        attended_features, _ = self.face_attention(
            query=human_query.expand(hidden_states.shape[0], -1, -1),
            key=hidden_states,
            value=hidden_states,
        )

        # Apply emotion enhancement
        enhancement = self.emotion_modules[module_idx](attended_features)

        # Mix with original features using adaptation factor
        enhanced_states = hidden_states + self.adaptation_factor * enhancement

        return enhanced_states


register("cogvideox-i2v", "lora", CogVideoXI2VLoraTrainer)
