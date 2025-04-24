CKPT_PATH="data/experiments/image/pusht/diffusion_policy_transformer/train_0/epoch=0100-test_mean_score=0.748.ckpt"
#CKPT_PATH="outputs/2025-01-25/17-08-16/checkpoints/epoch=0100-test_mean_score=0.184.ckpt"

DEVICE="cuda:4"

# [Orange]
# a. HDFree VIS
#CKPT_PATH="data/outputs/2025.01.28/12.18.13_train_diffusion_transformer_hybrid_pusht_image/checkpoints/epoch=0140-test_mean_score=0.281.ckpt"
# b. HDFree ALL
#CKPT_PATH="data/outputs/2025.01.28/12.14.31_train_diffusion_transformer_hybrid_pusht_image/train_2/checkpoints/epoch=0520-test_mean_score=0.307.ckpt"
#CKPT_PATH="data/outputs/2025.01.29/14.37.34_train_diffusion_transformer_hybrid_pusht_image/train_1/checkpoints/epoch=0070-test_mean_score=0.277.ckpt"
#CKPT_PATH="2025.01.29/14.37.34_train_diffusion_transformer_hybrid_pusht_image/no_tested"

# [Texture]
# a. HDFree VIS

# b. HDFree ALL
#CKPT_PATH="data/outputs/2025.01.29/12.51.39_train_diffusion_transformer_hybrid_pusht_image/train_0/checkpoints/epoch=0055-test_mean_score=0.288.ckpt"


# [Size]
# a. HDFree VIS
CKPT_PATH="data/outputs/2025.01.29/17.05.40_train_diffusion_transformer_hybrid_pusht_image/checkpoints/epoch=0040-test_mean_score=0.522.ckpt"
# b. HDFree ALL
#CKPT_PATH="data/outputs/2025.01.29/16.23.19_train_diffusion_transformer_hybrid_pusht_image/checkpoints/epoch=0120-test_mean_score=0.539.ckpt"
#CKPT_PATH="data/outputs/2025.01.29/19.20.11_train_diffusion_transformer_hybrid_pusht_image/train_0/checkpoints/epoch=0130-test_mean_score=0.561.ckpt"
#CKPT_PATH="data/outputs/2025.01.29/19.20.11_train_diffusion_transformer_hybrid_pusht_image/train_2/checkpoints/epoch=0110-test_mean_score=0.503.ckpt "



OUT_DIR="data/pusht_eval_output"
FROM_CONFIG="./image_pusht_diffusion_policy_transformer.yaml"

set -e
set -x

#yes | python eval.py -c ${CKPT_PATH} -o ${OUT_DIR} -d ${DEVICE} -f ${FROM_CONFIG}

yes | python eval.py -c ${CKPT_PATH} -o ${OUT_DIR} -d ${DEVICE}
