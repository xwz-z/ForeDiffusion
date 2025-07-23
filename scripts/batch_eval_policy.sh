# for example
# bash scripts/batch_eval.sh dp3 adroit 0 0 index
cd ForeDiffusion

# 1 alg_name; 2 class; 3 seed; 4 gpu; 5 index;

DEBUG=False

index=${5}
# adroit dexart metaworld
class=${2}
prefix="${class}_"

# Adroit
# tasks=("hammer" "door" "pen")

# Dexart
# tasks=("laptop" "faucet" "toilet" "bucket")

# metaworld
# easy
# tasks=("button-press" "button-press-topdown" "button-press-topdown-wall" "button-press-wall" "coffee-button" "dial-turn" "door-close" "door-lock" "door-open" "door-unlock" "drawer-close" "drawer-open" "faucet-close" "faucet-open" "handle-press" "handle-pull" "handle-pull-side" "lever-pull" "plate-slide" "plate-slide-back" "plate-slide-back-side" "plate-slide-side" "reach" "reach-wall" "window-close" "window-open" "peg-unplug-side")

# medium
# tasks=("basketball" "bin-picking" "box-close" "coffee-pull" "coffee-push" "hammer" "peg-insert-side" "push-wall" "soccer" "sweep" "sweep-into")    

# hard
# tasks=("assembly" "hand-insert" "pick-out-of-hole" "pick-place" "push" "push-back")

# very hard
# tasks=("shelf-place" "disassemble" "stick-pull" "stick-push" "pick-place-wall")

tasks=("push")
mkdir -p val_logs
for idx in "${!tasks[@]}"; do
    # echo "${tasks[$idx]}"
    task_name=${tasks[$idx]}

    alg_name=${1}
    task_name=${prefix}${tasks[$idx]}
    config_name=${alg_name}
    addition_info=${index}
    seed=${3}
    exp_name=${task_name}-${alg_name}-${addition_info}
    run_dir="data/outputs/${exp_name}_seed${seed}"


    # gpu_id=$(bash scripts/find_gpu.sh)
    gpu_id=${4}

    export HYDRA_FULL_ERROR=1 
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    
    python eval.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}
                                    
                                    
    index=$((index + 1))
done

