import torch
import torch.nn as nn
from diffusers import CogVideoXTransformer3DModel
from typing import Dict, List, Optional, Tuple, Union

class EmotionAdapter(nn.Module):
    """
    Adapter for enhancing emotional expressions while preserving identity features.
    Uses layer-specific adaptation for the 42-layer CogVideoX-5B model.
    """
    def __init__(
        self,
        transformer: CogVideoXTransformer3DModel,
        emotion_rank: int = 16,
        emotion_alpha: float = 8.0,
        # Middle layers (15-30) focus on emotional features
        emotion_layers: List[int] = list(range(15, 31)),
        # Early layers (0-14) and late layers (31-41) preserve identity
        identity_layers: List[int] = list(range(0, 15)) + list(range(31, 42)),
    ):
        super().__init__()
        self.transformer = transformer
        self.emotion_rank = emotion_rank
        self.emotion_alpha = emotion_alpha
        self.emotion_layers = emotion_layers
        self.identity_layers = identity_layers
        
        # Create emotion adapters for selected layers
        self.emotion_down_projs = nn.ModuleDict()
        self.emotion_up_projs = nn.ModuleDict()
        
        for layer_idx in emotion_layers:
            hidden_size = transformer.transformer_blocks[layer_idx].attn1.to_q.in_features
            
            # Down projection reduces dimension for emotion-specific features
            self.emotion_down_projs[f"layer_{layer_idx}"] = nn.Linear(
                hidden_size, emotion_rank, bias=False
            )
            
            # Up projection expands back to original dimension with emotion enhancement
            self.emotion_up_projs[f"layer_{layer_idx}"] = nn.Linear(
                emotion_rank, hidden_size, bias=False
            )
            
            # Initialize with small weights for stable training
            nn.init.normal_(self.emotion_down_projs[f"layer_{layer_idx}"].weight, std=0.02)
            nn.init.zeros_(self.emotion_up_projs[f"layer_{layer_idx}"].weight)
    
    def forward(self, hidden_states, layer_idx):
        """Apply emotion adaptation to specific transformer layers"""
        if layer_idx in self.emotion_layers:
            layer_key = f"layer_{layer_idx}"
            
            # Apply emotion adaptation with scaling
            emotion_res = self.emotion_up_projs[layer_key](
                self.emotion_down_projs[layer_key](hidden_states)
            )
            
            # Scale the emotion residual by alpha/rank for stable training
            emotion_res = emotion_res * (self.emotion_alpha / self.emotion_rank)
            
            # Add enhanced emotion features to original hidden states
            return hidden_states + emotion_res
        
        return hidden_states