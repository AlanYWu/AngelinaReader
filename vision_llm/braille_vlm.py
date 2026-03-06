"""Braille VLM: LLaVA-style fusion of RetinaNet FPN features with Qwen3-8B.

Architecture:
    Image → frozen RetinaNet FPN → VisualTokenAdapter (trainable, 17.8M) →
    192 visual tokens (4096-dim) → injected into Qwen3-8B embedding sequence →
    autoregressive braille transcription.

The <image> placeholder in the input token sequence is expanded to the 192
visual tokens produced by the adapter.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import VisualFeatureConfig
from .feature_extractor import RetinaNetFeatureExtractor
from .visual_token_adapter import VisualTokenAdapter


IMAGE_PLACEHOLDER = "<image>"


class BrailleVLM(nn.Module):
    """Combined vision-language model for braille OCR.

    Components:
        1. Frozen RetinaNet FPN feature extractor (0 trainable params)
        2. Trainable VisualTokenAdapter (17.8M params)
        3. Qwen3-8B LLM (frozen or LoRA-tuned)

    The forward pass:
        1. Extract FPN features from image
        2. Project to 192 visual tokens via adapter
        3. Tokenize text, find <image> placeholder position
        4. Replace placeholder embedding with visual tokens
        5. Run through Qwen3 for next-token prediction
    """

    def __init__(
        self,
        config: VisualFeatureConfig,
        llm_name_or_path: str = "Qwen/Qwen3-8B",
        freeze_llm: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.config = config

        # Vision components
        self.feature_extractor = RetinaNetFeatureExtractor(config)
        self.adapter = VisualTokenAdapter(config)

        # LLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name_or_path, trust_remote_code=True
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

        # Add <image> as a special token if not already present
        if IMAGE_PLACEHOLDER not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [IMAGE_PLACEHOLDER]}
            )
            self.llm.resize_token_embeddings(len(self.tokenizer))

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

    def get_trainable_params(self):
        """Return dict of parameter groups and their counts."""
        adapter_params = sum(
            p.numel() for p in self.adapter.parameters() if p.requires_grad
        )
        llm_params = sum(
            p.numel() for p in self.llm.parameters() if p.requires_grad
        )
        return {"adapter": adapter_params, "llm": llm_params, "total": adapter_params + llm_params}

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Image → visual tokens.

        Args:
            images: (B, 3, H, W) normalized float tensor.

        Returns:
            (B, num_visual_tokens, llm_hidden_dim) visual token embeddings.
        """
        with torch.no_grad():
            fpn_features = self.feature_extractor(images)
        visual_tokens = self.adapter(fpn_features)
        return visual_tokens

    def _build_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """Replace <image> placeholder tokens with visual token embeddings.

        Args:
            input_ids: (B, L) token ids with <image> placeholder(s).
            attention_mask: (B, L) attention mask.
            visual_tokens: (B, V, D) visual embeddings from adapter.
            labels: (B, L) optional labels for loss computation.

        Returns:
            inputs_embeds: (B, L', D) with placeholders replaced by visual tokens.
            attention_mask: (B, L') expanded attention mask.
            labels: (B, L') expanded labels (if provided).
        """
        B, V, D = visual_tokens.shape
        embed_layer = self.llm.get_input_embeddings()

        all_embeds = []
        all_masks = []
        all_labels = [] if labels is not None else None

        for i in range(B):
            ids = input_ids[i]
            mask = attention_mask[i]
            lbl = labels[i] if labels is not None else None

            # Find <image> token position
            image_positions = (ids == self.image_token_id).nonzero(as_tuple=True)[0]

            if len(image_positions) == 0:
                # No placeholder — just embed normally
                all_embeds.append(embed_layer(ids))
                all_masks.append(mask)
                if all_labels is not None:
                    all_labels.append(lbl)
                continue

            # Split around the first <image> token
            pos = image_positions[0].item()

            # Embed text tokens before and after <image>
            pre_embeds = embed_layer(ids[:pos])  # (pos, D)
            post_embeds = embed_layer(ids[pos + 1:])  # (L-pos-1, D)

            # Concatenate: [pre_text] [visual_tokens] [post_text]
            combined = torch.cat(
                [pre_embeds, visual_tokens[i], post_embeds], dim=0
            )
            all_embeds.append(combined)

            # Expand attention mask
            pre_mask = mask[:pos]
            post_mask = mask[pos + 1:]
            vis_mask = torch.ones(V, dtype=mask.dtype, device=mask.device)
            all_masks.append(torch.cat([pre_mask, vis_mask, post_mask]))

            # Expand labels: visual token positions get -100 (ignored in loss)
            if all_labels is not None:
                pre_lbl = lbl[:pos]
                post_lbl = lbl[pos + 1:]
                vis_lbl = torch.full(
                    (V,), -100, dtype=lbl.dtype, device=lbl.device
                )
                all_labels.append(torch.cat([pre_lbl, vis_lbl, post_lbl]))

        # Pad to same length
        max_len = max(e.shape[0] for e in all_embeds)
        padded_embeds = torch.zeros(B, max_len, D, dtype=all_embeds[0].dtype, device=all_embeds[0].device)
        padded_masks = torch.zeros(B, max_len, dtype=all_masks[0].dtype, device=all_masks[0].device)
        padded_labels = None
        if all_labels is not None:
            padded_labels = torch.full(
                (B, max_len), -100, dtype=all_labels[0].dtype, device=all_labels[0].device
            )

        for i in range(B):
            L = all_embeds[i].shape[0]
            padded_embeds[i, :L] = all_embeds[i]
            padded_masks[i, :L] = all_masks[i]
            if padded_labels is not None:
                padded_labels[i, :L] = all_labels[i]

        return padded_embeds, padded_masks, padded_labels

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """Full forward pass: image + text → LLM output.

        Args:
            images: (B, 3, H, W) preprocessed braille images.
            input_ids: (B, L) tokenized prompt with <image> placeholder.
            attention_mask: (B, L) attention mask.
            labels: (B, L) target token ids for loss (-100 for ignored positions).

        Returns:
            transformers CausalLMOutput with loss (if labels provided) and logits.
        """
        visual_tokens = self.encode_image(images)

        inputs_embeds, attention_mask, labels = self._build_inputs(
            input_ids, attention_mask, visual_tokens, labels
        )

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate braille transcription from image.

        Args:
            images: (B, 3, H, W) preprocessed braille images.
            input_ids: (B, L) tokenized prompt with <image> placeholder.
            attention_mask: (B, L) attention mask.
            max_new_tokens: max tokens to generate.

        Returns:
            Generated token ids.
        """
        visual_tokens = self.encode_image(images)
        inputs_embeds, attention_mask, _ = self._build_inputs(
            input_ids, attention_mask, visual_tokens
        )

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
