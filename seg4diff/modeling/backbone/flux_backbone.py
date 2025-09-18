import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import FluxPipeline, AutoencoderKL
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps, calculate_shift

from diffusers.utils.torch_utils import randn_tensor
from detectron2.modeling import BACKBONE_REGISTRY, Backbone

@BACKBONE_REGISTRY.register()
class FluxBackbone(Backbone):
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

        # self.model_name = "black-forest-labs/FLUX.1-schnell"
        self.model_name = "black-forest-labs/FLUX.1-dev" # use guidance-distilled version of Flux

        self.pipe = FluxPipeline.from_pretrained( # use guidance-distilled version of Flux
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
            )
        # to run on low vram GPUs (i.e. between 4 and 32 GB VRAM)
        # self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

        self.use_lora = False
        
        # self.pipe.to(torch.float16) # casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once

        self.vae = self.pipe.vae
        self.transformer = self.pipe.transformer
        
        self.text_encoder = self.pipe.text_encoder_2
        self.text_encoder_2 = self.pipe.text_encoder
        
        self.tokenizer = self.pipe.tokenizer_2
        self.tokenizer_2 = self.pipe.tokenizer
        
        self.scheduler = self.pipe.scheduler
        self.inference_steps = 50
        # self.scheduler.config.use_dynamic_shifting = False
        # timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, self.inference_steps)
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.softmax = cfg.MODEL.BACKBONE.SOFTMAX
        self.softmax_temps = cfg.MODEL.BACKBONE.SOFTMAX_TEMPS
        if input_shape is not None:
            cache_attentions = input_shape
        
        # cache all layers' attention maps
        # cache_attentions = torch.arange(19)
        # single_cache_attentions = torch.arange(38)
            
        for l, blk in enumerate(self.transformer.transformer_blocks):
            blk.layer_id = l
            if cache_attentions is not None and l in cache_attentions:
                blk.attn.processor.attn_cache = []
                blk.attn.processor.q = num_tokens
                blk.attn.processor.softmax = self.softmax
                blk.attn.processor.head = self.head
                blk.attn.processor.keep_head = cfg.MODEL.BACKBONE.KEEP_HEAD
            else:
                blk.attn.processor.attn_cache = None
        
        for l, blk in enumerate(self.transformer.single_transformer_blocks):
            blk.layer_id = l + 19
            if cache_attentions is not None and l+19 in cache_attentions:
                blk.attn.processor.attn_cache = []
                blk.attn.processor.q = num_tokens
                blk.attn.processor.softmax = self.softmax
                blk.attn.processor.head = self.head
                blk.attn.processor.keep_head = cfg.MODEL.BACKBONE.KEEP_HEAD
            else:
                blk.attn.processor.attn_cache = None
        
        if num_tokens > 0:
            self.prompt_tokens = nn.Parameter(torch.zeros(1, num_tokens, 4096), requires_grad=True)
        else:
            self.prompt_tokens = None
        
        self.texts = None
        self.prompt_embeds = None
        self.prompt_embeds_list = None
        self.pooled_prompt_embeds = None
        
        self.with_sos_eos = None
        
        # for viz save
        self.image_name = None
        
        print("FluxBackbone initialized.")
                
    def forward(self, x, prompt=None, noise=None, noise_steps=None, prompt_embeds=None, get_class_logits=False, mean_heads=False, selected_idxs=None, with_sos_eos=False, indices=None):
        """
            Simplified forward pass for SD3 backbone.
            
            Args:
                x: torch.Tensor of shape (N, C, H, W)
                text: torch.Tensor of shape (N, L)
                noise: torch.Tensor of shape (N, L, D)
        """
        device = x.device
        self.with_sos_eos = with_sos_eos

        # define latent input shape
        num_channels_latents = self.transformer.config.in_channels // 4
        height = 2 * (int(x.shape[2]) // self.pipe.vae_scale_factor)
        width = 2 * (int(x.shape[3]) // self.pipe.vae_scale_factor)
        
        shape = (x.shape[0], num_channels_latents, height, width) # shape of latent_model_input
        
        # set timestep
        sigmas = np.linspace(1.0, 1 / self.inference_steps, self.inference_steps)
        image_seq_len = shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            self.inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        timestep = timesteps[-noise_steps].expand(shape[0])
        
        # prepare prompt embeddings
        if prompt is not None :
            prompt_embeds, pooled_prompt_embeds, text_ids = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                max_sequence_length=self.pipe.tokenizer_max_length,
                device=device,
            )
        
        # for viz save
        # for l, blk in enumerate(self.transformer.transformer_blocks):
        #     blk.image_name = self.image_name
        
        # prepare latent input
        latent = self.image_to_latents(x).to(device=device)
        
        if noise is None and noise_steps is None:
            latent_model_input = latent
        else:
            noise = noise if noise is not None else randn_tensor(shape, device=device, generator=self.generator) #, dtype=latent.dtype)
        
        latent = self.scheduler.scale_noise(latent, self.scheduler.timesteps[-noise_steps].unsqueeze(0), noise)
        latent_model_input = self.pipe._pack_latents(latent, x.shape[0], num_channels_latents, height, width)
        
        latent_image_ids = self.pipe._prepare_latent_image_ids(x.shape[0], height, width, device, x.dtype)
        
        
        # guidance
        guidance = torch.full([1], 3.5, device=device, dtype=torch.float16)
        guidance = guidance.expand(shape[0])

        guidance = None if "schnell" in self.model_name else guidance

        # forward pass
        num_tokens = len(self.pipe.tokenizer(prompt)["input_ids"]) if not self.softmax else None
        indices = None if self.softmax else indices
        pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep/1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                    # num_tokens=num_tokens,
                    softmax_weights=None if self.softmax else self.softmax_temps,
                    indices=indices,
                    return_aux=False,
        )
        
        if indices is not None:
            return pred
        
        out = []

        for l, blk in enumerate(self.transformer.transformer_blocks):
            if blk.attn.processor.attn_cache is not None:
                out.append(blk.attn.processor.attn_cache)
                blk.attn.processor.attn_cache = []
        
        for l, blk in enumerate(self.transformer.single_transformer_blocks):
            if blk.attn.processor.attn_cache is not None:
                out.append(blk.attn.processor.attn_cache)
                blk.attn.processor.attn_cache = []
                
        # if mean_heads:
        #     import pdb; pdb.set_trace()
        #     out = [o.mean(dim=1, keepdim=True) for o in out]
            #cls_out = [o.mean(dim=1, keepdim=True) for o in cls_out]
            
        pred = torch.stack(out, dim=1)
        return {
            'attn_cache': pred,
        }

    def image_to_latents(self, image: torch.Tensor):
        
        posterior = self.vae.encode(image).latent_dist.mean

        image_latents = (posterior - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents
