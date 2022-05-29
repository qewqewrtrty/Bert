basedir=./
export PYTHONPATH=./bert
train_data=./Test/data/finetune/sts_b.train
test_data=./Test/data/finetune/sts_b.test
vocab=./Test/data/pretrain/vocab
output=./Test/output.finetune.1
pretrained_model=./Test/output.1.bertlm.epoch.7.end

python $basedir/bert/runbert/finetune.py -i gbk \
    -c $train_data \
    -t $test_data \
    -v $vocab \
    -o $output \
    -b 128 \
    -s 42 -l 6 -a 16 -hs 256 \
    --log_freq 10 --lr 0.0001 --save_freq 10 --cuda_devices 2 \
    --adam_weight_decay 0.0 -s 42 --num_workers 2 \
    --on_memory \
    -p $pretrained_model

