

# ForeDiffusion: A Foresight-Conditioned Diffusion Policy via Future View


![[fig/Homepage.png]]
**The Foresight-Conditioned Diffusion (ForeDiffusion)** is introduced by injecting the predicted future view representation into the diffusion process, and the strategy is guided to be forward-looking, thereby correcting the trajectory deviation. Following this design, ForeDiffusion employs a dual loss optimization, combining the traditional denoising loss and the consistency loss of future observations, to achieve the unified optimization of local action accuracy and overall task temporal coherence. Extensive evaluation on the MetaWorld benchmark and the Adroit suite demonstrates that ForeDiffusion achieves an average success rate of 80\% for the overall task, significantly outperforming the existing mainstream diffusion methods in high difficulty tasks, while maintaining stable performance across the entire task set.

## ✨ Insights

- Considering the importance of global conditioning in the diffusion process, enriching the conditioning information can potentially lead to more accurate and stable trajectory predictions.
- Introducing new conditioning signals necessitates designing corresponding new loss functions for effective self-supervised training.

# Installation

## 🏠 Clone Project
```bash
git clone
cd ForeDiffusion
```
---
## 🛠️ Install Conda Environment

1. Create Python Environment
```bash
conda deactivate
conda env remove -n ForeDiffusion
conda create -n ForeDiffusion python=3.8
conda activate ForeDiffusion 
```

2. Install torch
```bash
# CUDA 12.2 by default, If your system has CUDA >= 12.1: 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

\# For other CUDA versions, please refer to the official PyTorch installation guide to find the appropriate command.
```
3. Install ForeDiffusion
```bash
pip install -e .
>pip install zarr\==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core\==1.2.0 dill\==0.3.5.1 einops\==0.4.1 diffusers\==0.11.1 numba\==0.56.4 moviepy imageio av matplotlib termcolor
```

4. Install pytorch3d and visualizer (References the [3DP Repository](https://github.com/YanjieZe/3D-Diffusion-Policy).)
```bash
cd third_party/pytorch3d_simplified && pip install -e . && cd ..

pip install kaleido plotly
cd visualizer && pip install -e . && cd ..
```
---
## 🔎 Simulation

The simulation environment setup follows the instructions detailed in [Simulation](Simulation.md). It covers the installation and configuration of MuJoCo, necessary Python dependencies, third-party libraries, and required assets.

# Benchmarks of ForeDiffusion
You could generate demonstrations by yourself using our provided expert policies.  Generated demonstrations are under `$YOUR_REPO_PATH/3D-Diffusion-Policy/data/`.
## Adroit
Download expert demonstrations on the **Adroit** benchmark from [OneDrive](https://1drv.ms/u/s!Ag5QsBIFtRnTlFWqYWtS2wMMPKNX?e=dw8hsS), and place the extracted `ckpts` folder under: `$YOUR_REPO_PATH/third_party/VRL3/`.
```bash
bash scripts/gen_demonstration_adroit.sh {Task Name}
```
## MetaWorld
Generate expert demonstrations on the **MetaWorld** benchmark, simply run:
```bash
bash scripts/gen_demonstration_metaworld.sh {Task Name}
```
Make sure all required assets and dependencies are properly set up before running the script.

⚠️ Disclaimer & Demonstration Accuracy Notice: Expert demonstrations are generated using our trained policies, which aim to closely replicate ground-truth expert behavior. However, due to stochasticity in policy execution or environmental randomness, **a small margin of error is acceptable**.
If a generated demonstration exhibits significant deviation from the expected expert behavior, we recommend **re-generating it manually** to ensure data quality.

# ⚙️ Usage

Make sure the environment has been correctly set up, expert demonstrations have been successfully generated, and `wandb` has been logged in to allow real-time monitoring and experiment tracking.

We provide scripts under the `scripts` folder for both single-task and batch-task execution. The overall workflow is as follows:
1. Expert Demonstration Generation
A batch script is provided to generate expert demonstrations for MetaWorld tasks: 
```bash
# bash scripts/batch_gen.sh {Task Class} {index}
bash scripts/batch_gen.sh adroit 0
```
The generated data will be saved under the `data/` directory.

2. Training
Scripts are available for both single-task and batch-task training: 
```bash
# Single-task
# bash scripts/train_policy.sh ForeDiffusion {Task Class}_{Task Name}} {Index} {Seed} {GPU}
bash scripts/train_policy.sh ForeDiffusion adroit_hammer 0 0 0

# Batch-task
# bash scripts/batch_train.sh ForeDiffusion {Task Class} {Seed} {GPU} {index}
bash scripts/batch_train.sh ForeDiffusion adroit 0 0 0
```
The model will be trained following the provided task `.yaml` configurations, for 3000 epochs by default. The best performing model (highest success rate) and the final model weights will be saved.

3. Evaluation
A script is provided to evaluate trained models: 
```bash
# Single-task
# bash scripts/eval_policy.sh ForeDiffusion {Task Class}_{Task Name}} {Index} {Seed} {GPU}
bash scripts/eval_policy.sh ForeDiffusion adroit_hammer 0 0 0

# Batch-task
# bash scripts/batch_eval.sh ForeDiffusion {Task Class} {Seed} {GPU} {index}
bash scripts/batch_eval.sh ForeDiffusion adroit 0 0 0
```
Evaluation includes visualization of the agent performance. All results will be logged to the corresponding `wandb` project.

# 📈 Baselines

[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
[3D Diffusion Policy](https://3d-diffusion-policy.github.io/), 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations
[FlowPolicy](https://github.com/zql-kk/FlowPolicy), FlowPolicy: Enabling Fast and Robust 3D Flow-based Policy via Consistency Flow Matching for Robot Manipulation
[SDM Policy](https://sdm-policy.github.io/), Score and Distribution Matching Policy: Advanced Accelerated Visuomotor Policies via Matched Distillation
[ManiCM](https://manicm-fast.github.io/), ManiCM: Real-time 3D Diffusion Policy via Consistency Model for Robotic Manipulation

![[fig/Result.png]]
# License

# Acknowledge
Our code is generally built upon: [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy), [VRL3](https://github.com/microsoft/VRL3), [MetaWorld](https://github.com/Farama-Foundation/Metaworld),[Crossway Diffusion](https://github.com/LostXine/crossway_diffusion). We thank all these authors for their nicely open sourced code and their great contributions to the community.
# Citation
If you find our work useful, please consider citing:

