CUDA_VISIBLE_DEVICES=0,1 \
    python -m torch.distributed.launch --nproc_per_node=2 \
    train_lfd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --lambda-kd 1.0 \
    --batch-size 16 \
    --val-per-iters 500 \
    --save-per-iters 500 \
    --data /data/cityscape/ \
    --lfd-start-iter 1000 \
    --lambda-lfd 1. \
    --save-dir LFD_checkpoint \
    --log-dir LFD_log \
    --teacher-pretrained /home/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base /home/resnet18-imagenet.pth


CUDA_VISIBLE_DEVICES=0,1 \
    python -m torch.distributed.launch --nproc_per_node=2 \
    train_lfd.py \
    --teacher-model deeplabv3 \
    --student-model deeplab_mobile \
    --teacher-backbone resnet101 \
    --student-backbone mobilenetv2 \
    --lambda-kd 1.0 \
    --batch-size 16 \
    --val-per-iters 500 \
    --save-per-iters 500 \
    --lfd-start-iter 1000 \
    --lambda-lfd 1. \
    --data /data/cityscape/ \
    --save-dir LFD_checkpoint \
    --log-dir LFD_log \
    --teacher-pretrained /home/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base /home/mobilenetv2-imagenet.pth

