LLaVA_HOME=/code/LLaVA
LLAVA_PATH=$LLAVA_HOME/scripts/v1_5/eval
RESULT_PATH=/userhome/result

CUDA_VISIBLE_DEVICES=0 nohup bash $LLAVA_PATH/mmbench.sh $1 test > $RESULT_PATH/$1/mmbench.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash $LLAVA_PATH/mmbench_cn.sh $1 test > $RESULT_PATH/$1/mmbench_cn.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash $LLAVA_PATH/mme.sh $1 > $RESULT_PATH/$1/mme.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash $LLAVA_PATH/mmvet.sh $1 > $RESULT_PATH/$1/mmvet.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup bash $LLAVA_PATH/pope.sh $1 > $RESULT_PATH/$1/pope.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup bash $LLAVA_PATH/sqa.sh $1 > $RESULT_PATH/$1/sqa.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup bash $LLAVA_PATH/textvqa.sh $1 > $RESULT_PATH/$1/textvqa.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup bash $LLAVA_PATH/vizwiz.sh $1 > $RESULT_PATH/$1/vizwiz.log 2>&1 &

wait
cd $LLaVA_HOME
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash $LLAVA_PATH/gqa.sh $1 > $RESULT_PATH/$1/gqa.log 2>&1
cd $LLaVA_HOME
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash $LLAVA_PATH/seed.sh $1 > $RESULT_PATH/$1/seed.log 2>&1
cd $LLaVA_HOME
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash $LLAVA_PATH/vqav2.sh $1 > $RESULT_PATH/$1/vqav2.log 2>&1