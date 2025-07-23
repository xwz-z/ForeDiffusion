## Mujoco
>cd ~/.mujoco
>wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate
>tar -xvzf mujoco210.tar.gz


>export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\${HOME}/.mujoco/mujoco210/binr.gz -O
>export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
>export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
>export MUJOCO_GL=egl

>cd YOUR_PATH_TO_THIRD_PARTY
>cd mujoco-py-2.1.2.14
>pip install -e .
>cd ../..


## Install Environment

>pip install setuptools\==59.5.0 Cython\==0.29.35 patchelf\==0.17.2.0
>
>cd third_party
>cd dexart-release && pip install -e . && cd ..
>cd gym-0.21.0 && pip install -e . && cd ..
>cd Metaworld && pip install -e . && cd ..
>cd rrl-dependencies && pip install -e mj_envs/. && pip install -e mjrl/. && cd ..

download assets from [Google Drive](https://drive.google.com/file/d/1DxRfB4087PeM3Aejd6cR-RQVgOKdNrL4/view?usp=sharing), unzip it, and put it in `third_party/dexart-release/assets`. 

download Adroit RL experts from [OneDrive](https://1drv.ms/u/s!Ag5QsBIFtRnTlFWqYWtS2wMMPKNX?e=dw8hsS), unzip it, and put the `ckpts` folder under `$YOUR_REPO_PATH/third_party/VRL3/`.