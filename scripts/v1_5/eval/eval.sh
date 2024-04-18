LLAVA_PATH=./scripts/v1_5/eval

CUDA_VISIBLE_DEVICES=0 nohup bash $LLAVA_PATH/mmbench.sh $1 test > $LLAVA_PATH/mmbench.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash $LLAVA_PATH/mmbench_cn.sh $1 test > $LLAVA_PATH/mmbench_cn.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash $LLAVA_PATH/mme.sh $1 > $LLAVA_PATH/mme.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash $LLAVA_PATH/mmvet.sh $1 > $LLAVA_PATH/mmvet.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup bash $LLAVA_PATH/pope.sh $1 > $LLAVA_PATH/pope.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup bash $LLAVA_PATH/sqa.sh $1 > $LLAVA_PATH/sqa.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup bash $LLAVA_PATH/textvqa.sh $1 > $LLAVA_PATH/textvqa.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup bash $LLAVA_PATH/vizwiz.sh $1 > $LLAVA_PATH/vizwiz.log 2>&1 &

wait
cd /home/lanxy/LLaVA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash $LLAVA_PATH/gqa.sh $1 > $LLAVA_PATH/gqa.log 2>&1
cd /home/lanxy/LLaVA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash $LLAVA_PATH/seed.sh $1 > $LLAVA_PATH/seed.log 2>&1
cd /home/lanxy/LLaVA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash $LLAVA_PATH/vqav2.sh $1 > $LLAVA_PATH/vqav2.log 2>&1