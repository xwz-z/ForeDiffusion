from foresight_conditioned.model.diffusion.conditional_unet1d import *
from foresight_conditioned.model.diffusion.conv2d_components import (Upsample2d, Conv2dBlock)

class FVDP(ConditionalUnet1D):
    def __init__(self, 
        input_dim,
        n_obs_steps: int,
        obs_shape_meta: dict,
        decode_unet_feat=True,
        decode_pe_dim=64,
        decode_resolution=2,
        decode_dims=[64, 128],
        decode_low_dim_dims=[4, 2],
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        condition_type='film',
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        cond_predict_scale=False,
        ):
        super().__init__(
            input_dim,
            local_cond_dim,
            global_cond_dim,
            diffusion_step_embed_dim,
            down_dims,
            kernel_size,
            n_groups,
            condition_type,
        )
        all_dims = [input_dim] + list(decode_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        decode_low_dim_dims.append(1)
        
        self.obs_shape_meta = obs_shape_meta
        self.n_obs_steps = n_obs_steps
        decs = {}
        
        self.decode_unet_feat = decode_unet_feat
        # if this is true, decode the output from the mid module of unet
        # else, decode global cond
        if decode_unet_feat:
            dec_chn = down_dims[-1] // self.n_obs_steps // decode_resolution // decode_resolution 
            assert dec_chn == decode_dims[-1], f'The decoder dim must match the dim of UNet + PE ({dec_chn} vs {decode_dims[-1]})'
            self.decode_resolution = decode_resolution
            self.decode_pe_dim = decode_pe_dim
        else:
            raise NotImplementedError('Haven\'t finished yet.')
        
        for key, obs in obs_shape_meta.items():
            if obs['type'] == 'point_cloud' or 'rgb':
                # if crop_shape is not None:
                #    self.rgb_shape = crop_shape
                # else:
                if 'recon_shape' in obs:
                    self.rgb_shape = obs['recon_shape'][1:]
                else:
                    self.rgb_shape = obs['shape'][1:] # C H W => H W
                    
                start_dim = decode_dims[0]
                
                rgb_dec = nn.ModuleList([])
                for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
                    is_last = ind >= (len(in_out) - 1)
                    rgb_dec.append(nn.Sequential(
                        ResidualBlock2D(
                            dim_out + decode_pe_dim, dim_in,
                            kernel_size=kernel_size, n_groups=n_groups),
                        ResidualBlock2D(
                            dim_in, dim_in, kernel_size=kernel_size, n_groups=n_groups),
                        Upsample2d(dim_in) if not is_last else nn.Identity()
                    ))
                
                final_conv = nn.Sequential(
                    Conv2dBlock(start_dim, start_dim, kernel_size=kernel_size),
                    nn.Conv2d(start_dim, obs['shape'][0], 1),
                )
                decs[key] = rgb_dec
                decs[key + '_final'] = final_conv
            
            elif obs['type'] == 'low_dim':
                low_dim_dims = list(zip(decode_low_dim_dims[:-1], decode_low_dim_dims[1:]))
                obs_dim = obs['shape'][0]
                ld_dec = [
                    nn.Linear(down_dims[-1] // self.n_obs_steps, obs_dim * decode_low_dim_dims[0]),
                ]
                for dim_in, dim_out in low_dim_dims:
                    ld_dec.append(nn.Mish())
                    ld_dec.append(nn.Linear(obs_dim * dim_in, obs_dim * dim_out))
                decs[key] = nn.ModuleList(ld_dec)
                
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        self.decs = nn.ModuleDict(decs)


        self.feature_predictor = GlobalConditionPredictor()
        self.pose_predictor = NextPosePredictor(1024)
        

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        timestep_embed = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            if self.condition_type == 'cross_attention':
                timestep_embed = timestep_embed.unsqueeze(1).expand(-1, global_cond.shape[1], -1)
            global_feature = torch.cat([timestep_embed, global_cond], axis=-1)  

        # predict
        pred_cond = self.feature_predictor(global_cond)
        global_feature = torch.cat([global_feature, pred_cond], axis=-1)  

        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            if self.use_down_condition:
                x = resnet(x, global_feature)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x)
            h.append(x)
            x = downsample(x)


        for mid_module in self.mid_modules:
            if self.use_mid_condition:
                x = mid_module(x, global_feature)
            else:
                x = mid_module(x)

        # # predict
        pred_cond = self.feature_predictor(global_cond)
        global_feature = torch.cat([global_feature, pred_cond], axis=-1)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            if self.use_up_condition:
                x = resnet(x, global_feature)
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x)
            x = upsample(x)


        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x
    
    def forward_w_mid(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        timestep_embed = self.diffusion_step_encoder(timesteps)
        # timestep_embed [128, 128]
        if global_cond is not None:
            if self.condition_type == 'cross_attention':
                timestep_embed = timestep_embed.unsqueeze(1).expand(-1, global_cond.shape[1], -1)
            global_feature = torch.cat([timestep_embed, global_cond], axis=-1)
        # concat timestep_embed and global_cond [128, 384]

        # recons = {}
        # x = self.feature_predictor(global_cond)
        # recons['obs'] = x
        # global_feature = torch.cat([global_feature, recons['obs']], axis=-1)

        recons = {}
        x = self.feature_predictor(global_cond)
        recons['obs'] = x
        global_feature = torch.cat([global_feature, recons['obs']], axis=-1)
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
            
        x = sample  # noisy_trajectory [128, 4, 16]
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            if self.use_down_condition:
                x = resnet(x, global_feature)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x)
            h.append(x)
            x = downsample(x)

        # [128, 2048, 4]
        for mid_module in self.mid_modules:
            if self.use_mid_condition:
                x = mid_module(x, global_feature)
            else:
                x = mid_module(x)
        mid = x.clone() # [128, 2048, 4]

        return mid, global_feature, h, h_local
    
    def forward_w_after(self, 
                        x, global_feature, h, h_local):
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            if self.use_up_condition:
                # print("resnet: ", resnet)
                x = resnet(x, global_feature)
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x
    
    def forward_w_pred(self,
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        local_cond=None, global_cond=None, **kwargs):
        mid, global_feature, h, h_local = self.forward_w_mid(sample=sample,
                                timestep=timestep,
                                local_cond=local_cond,
                                global_cond=global_cond)
        
        
        recons = {}
        x = self.feature_predictor(global_cond)
        recons['obs'] = x
        global_feature = torch.cat([global_feature, recons['obs']], axis=-1)
        
        x_res = self.forward_w_after(x=mid, global_feature=global_feature, h=h, h_local=h_local)
        
        return x_res, recons, mid
