# for example
# bash scripts/batch_gen.sh adroit index

DEBUG=False
save_ckpt=True
index=${2}
class=${1}

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

if [ "${class}" = "adroit" ]; then
    # adroit
    cd third_party/VRL3/src
elif [ "${class}" = "dexart" ]; then
    # dexart
    num_episodes=100
    root_dir=../../ForeDiffusion/data/
    cd third_party/dexart-release
elif [ "${class}" = "metaworld" ]; then
    # meta
    cd third_party/Metaworld
else
    echo "input error"
fi

for idx in "${!tasks[@]}"; do
    # echo "${tasks[$idx]}"
    task_name=${tasks[$idx]}

    CUDA_VISIBLE_DEVICES=1
    if  [ "${class}" = "adroit" ]; then
        # adroit
        CUDA_VISIBLE_DEVICES=0 python gen_demonstration_expert.py --env_name ${task_name} \
                        --num_episodes 10 \
                        --root_dir "../../../ForeDiffusion/data/" \
                        --expert_ckpt_path "../ckpts/vrl3_${task_name}.pt" \
                        --img_size 84 \
                        --not_use_multi_view \
                        --use_point_crop
    elif [ "${class}" = "dexart" ]; then
        # dexart
        python examples/gen_demonstration_expert.py --task_name=${task_name} \
                --checkpoint_path assets/rl_checkpoints/${task_name}/${task_name}_nopretrain_0.zip \
                --num_episodes $num_episodes \
                --root_dir $root_dir \
                --img_size 84 \
                --num_points 1024
    elif [ "${class}" = "metaworld" ]; then
        # meta
        python gen_demonstration_expert.py --env_name=${task_name} \
                    --num_episodes 10 \
                    --root_dir "../../ForeDiffusion/data/"
    else
        echo "input error"
    fi

    index=$((index + 1))
done
