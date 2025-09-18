import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import StableDiffusion3Pipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from peft import LoraConfig, set_peft_model_state_dict
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

@BACKBONE_REGISTRY.register()
class SD35Backbone(Backbone):
    def __init__(
      self,
      cfg,
      input_shape,
      cache_attentions=None,
    ):
        super().__init__()
        
        num_tokens = cfg.MODEL.BACKBONE.NUM_TOKENS
        force_fp16 = cfg.MODEL.BACKBONE.FP16
        
        self.layer = cfg.MODEL.LAYER
        self.head = cfg.MODEL.HEAD
        if self.head is not None:
            print("Using head", self.head)

        self.seed = cfg.MODEL.BACKBONE.SEED
        self.generator = torch.Generator(device="cuda").manual_seed(self.seed) if self.seed is not None else None
        
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium",
            text_encoder_3=None,
            torch_dtype=torch.float16 if force_fp16 else None,
            )
        self.pipe.to("cuda")

        self.vae = self.pipe.vae
        self.transformer = self.pipe.transformer
        
        self.text_encoder = self.pipe.text_encoder
        self.text_encoder_2 = self.pipe.text_encoder_2
        self.text_encoder_3 = self.pipe.text_encoder_3 #.to(torch.float16)
        
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer_2 = self.pipe.tokenizer_2
        self.tokenizer_3 = self.pipe.tokenizer_3
        
        self.use_lora = cfg.MODEL.BACKBONE.USE_LORA
        self.use_learnable_tokens = cfg.MODEL.BACKBONE.USE_LEARNABLE_TOKENS

        if self.use_lora:
            lora_layers = cfg.MODEL.BACKBONE.LORA_LAYERS
            lora_blocks = cfg.MODEL.BACKBONE.LORA_BLOCKS
            lora_rank = cfg.MODEL.BACKBONE.LORA_RANK

            if lora_blocks is not None:
                target_modules = [layer.strip() for layer in lora_blocks.split(",")]
            else:
                target_modules = [
                    # "attn.add_k_proj",
                    # "attn.add_q_proj",
                    # "attn.add_v_proj",
                    # "attn.to_add_out",
                    "attn.to_k",
                    "attn.to_q",
                    "attn.to_v",
                    "attn.to_out.0",
                ]
            if lora_layers is not None:
                target_blocks = lora_layers
                target_modules = [
                    f"transformer_blocks.{block}.{module}" for block in target_blocks for module in target_modules
                ]

            # now we will add new LoRA weights to the attention layers
            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            self.transformer.add_adapter(transformer_lora_config)
            for name, param in self.transformer.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            if cfg.MODEL.BACKBONE.LORA_WEIGHTS is not None:
                def preprocess_state_dict(state_dict, prefix_to_remove):
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith(prefix_to_remove): # remove prefix
                            k = k[len(prefix_to_remove):]
                        k = k.replace(".default.", ".") # remove .default.
                        new_state_dict[k] = v
                    return new_state_dict

                print("Loading LoRA weights from", cfg.MODEL.BACKBONE.LORA_WEIGHTS)
                state_dict = torch.load(cfg.MODEL.BACKBONE.LORA_WEIGHTS, map_location="cpu")

                prefix = "backbone.transformer."
                state_dict['model'] = preprocess_state_dict(state_dict['model'], prefix)
                outcome = set_peft_model_state_dict(self.transformer, state_dict['model'])

        # use self.scheduler only for validation
        self.scheduler = self.pipe.scheduler
        self.inference_steps = cfg.MODEL.NUM_INFERENCE_STEPS
        _, _ = retrieve_timesteps(self.scheduler, self.inference_steps)

        # use copy for training
        self.scheduler_copy = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            subfolder="scheduler",
            local_files_only=True
        )

        for param in self.parameters():
            param.requires_grad = False
        
        self.softmax = cfg.MODEL.BACKBONE.SOFTMAX
        self.softmax_temps = cfg.MODEL.BACKBONE.SOFTMAX_TEMPS
        self.keep_head = cfg.MODEL.BACKBONE.KEEP_HEAD
        self.max_head = cfg.MODEL.BACKBONE.MAX_HEAD
        self.softmax_i2t_only = cfg.MODEL.BACKBONE.SOFTMAX_I2T_ONLY
        print("Using softmax:", self.softmax, 
              "softmax_i2t_only:", self.softmax_i2t_only,
              "keep_head:", self.keep_head,
              "max_head:", self.max_head)

        if input_shape is not None:
            cache = input_shape
        for l, blk in enumerate(self.transformer.transformer_blocks):
            blk.layer_id = l
            blk.attn.processor.attn_cache = None
            if cache is not None and l in cache:
                blk.attn.processor.attn_cache = []
                # Args for attention cache
                blk.attn.processor.q = num_tokens
                blk.attn.processor.softmax = self.softmax
                blk.attn.processor.head = self.head
                blk.attn.processor.keep_head = self.keep_head
                blk.attn.processor.max_head = self.max_head
                blk.attn.processor.softmax_i2t_only = self.softmax_i2t_only

        self.prompt_tokens = None
        self.pooled_prompt_embeds = None
        
        self.texts = None
        self.prompt_embeds = None
        self.prompt_embeds_list = None
        
        # for viz save
        self.image_name = None

    def forward(self, x, prompt=None, noise=None, noise_steps=None, prompt_embeds=None, get_class_logits=False, mean_heads=False, selected_idxs=None, with_sos_eos=False, indices=None):
        """
            Simplified forward pass for SD3 backbone.
            
            Args:
                x: torch.Tensor of shape (N, C, H, W)
                text: torch.Tensor of shape (N, L)
                noise: torch.Tensor of shape (N, L, D)
        """
        device = x.device

        with torch.no_grad():
            # Prepare the latent code
            latent = img_to_latents(x, self.vae).to(device=device) #, dtype=x.dtype)
            latent_model_input, timestep, noise = self.prepare_latent(latent, noise=noise, noise_steps=noise_steps)
            
            if noise_steps is not None:
                timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, self.inference_steps, device)
                timestep = timesteps[-timestep].expand(latent_model_input.shape[0])
            else:
                timestep = timestep.expand(latent_model_input.shape[0])    
            
            # Prepare the text embeddings
            if prompt is not None :
                prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                    prompt=prompt,
                    prompt_2=None,
                    prompt_3=None,
                    do_classifier_free_guidance=False,
                    device=device,
                )

            # Clear the attention cache
            for l, blk in enumerate(self.transformer.transformer_blocks):
                if blk.attn.processor.attn_cache is not None:
                    blk.attn.processor.attn_cache = []
        # cache image name to transformer block for viz save
        # for l, blk in enumerate(self.transformer.transformer_blocks):
        #     blk.image_name = self.image_name

        num_tokens = len(self.pipe.tokenizer(prompt)["input_ids"]) if not self.softmax else None
        indices = None if self.softmax else indices
        # Predict the noise residual
        
        pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds if not self.use_learnable_tokens else self.prompt_tokens,
            pooled_projections=pooled_prompt_embeds if not self.use_learnable_tokens else self.pooled_prompt_embeds,
            return_dict=False,
        )[0]

        # if sigma is not None:
        #     pred = pred * (-sigma) + latent_model_input

        # if indices is not None:
        #     return pred
        
        out = []
        feat_sim = []
        feat = []

        if self.head == None and self.layer == None:
            for l, blk in enumerate(self.transformer.transformer_blocks):
                if blk.attn.processor.attn_cache is not None:
                    out.append(blk.attn.processor.attn_cache.to(device=device))
        else:
            for l, blk in enumerate(self.transformer.transformer_blocks):
                if l == self.layer:
                    if blk.attn.processor.attn_cache is not None:
                        out.append(blk.attn.processor.attn_cache.to(device=device))

        out = torch.stack(out, dim=1) if len(out) > 0 else None
        feat_sim = torch.stack(feat_sim, dim=1) if len(feat_sim) > 0 else None
        feat = torch.stack(feat, dim=1) if len(feat) > 0 else None
        # out = None
        return {
            "model_pred": pred,
            "noise": noise,
            "model_input": latent,
            "attn_cache": out,
            "timestep": timestep,
        }
            
    def invert_latent(self, latent, prompt, invert_steps):
        # Invert the latent code
        noise = randn_tensor(latent.shape, device=latent.device, dtype=latent.dtype)
        # FIX: use latent (not an undefined latent_model_input) as input to scale_noise.
        latent_model_input = self.pipe.scheduler.scale_noise(latent, self.pipe.scheduler.timesteps[-1].unsqueeze(0), noise)
        
        # Use a fixed guidance flag (same as in prepare_diffusion_inputs)
        do_classifier_free_guidance = False
        (
            inversion_prompt_embeds,
            negative_inversion_prompt_embeds,
            inversion_pooled_prompt_embeds,
            negative_inversion_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(prompt=prompt, prompt_2=prompt, prompt_3=prompt,
                                do_classifier_free_guidance=do_classifier_free_guidance)
        
        # latent_model_input = latent_model_input.to(dtype=torch.float16)
        
        for t in range(2, invert_steps+1):
            # Use the device of latent_model_input instead of hardcoding 'cuda'
            timestep = self.pipe.scheduler.timesteps[-t+1].expand(latent_model_input.shape[0]).to(latent_model_input.device)
        
            with torch.no_grad():
                noise_pred = self.pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=inversion_prompt_embeds,
                    pooled_projections=inversion_pooled_prompt_embeds,
                    return_dict=False,
                )[0]
        
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            
            latent_model_input = latent_model_input + (self.pipe.scheduler.sigmas[-t] - self.pipe.scheduler.sigmas[-t+1]) * noise_pred

        return latent_model_input

    def prepare_latent(self, latent, noise=None, noise_steps=None):
        
        if noise is None:
            noise = randn_tensor(latent.shape, device=latent.device)
        
        # Prepare the latent code
        if noise_steps is None:
            # training
            latent_model_input, noise_steps, _ = self.sample_noise(latent, noise, latent.shape[0])
        else:
            # inference
            latent_model_input = self.scheduler.scale_noise(latent, self.scheduler.timesteps[-noise_steps].unsqueeze(0), noise)
        
        return latent_model_input, noise_steps, noise

    def sample_noise(self, model_input, noise, bsz):
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            logit_mean=0.0,
            logit_std=0.5,
            mode_scale=1.29,
            batch_size=bsz,
        )
        indices = (u * self.scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.scheduler_copy.timesteps[indices].to(device=self.transformer.device)
        
        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = self.scheduler_copy.sigmas.to(device=self.transformer.device, dtype=dtype)
            schedule_timesteps = self.scheduler_copy.timesteps.to(self.transformer.device)
            timesteps = timesteps.to(self.transformer.device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma
        
        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        return noisy_model_input, timesteps, sigmas
    
def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = (posterior.mean - vae.config.shift_factor) * vae.config.scaling_factor
    
    return latents
