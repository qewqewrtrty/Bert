basedir=./
export PYTHONPATH=./bert
train_data=./Test/data/pretrain/corpus
test_data=./Test/data/pretrain/corpus.test
vocab=./Test/data/pretrain/vocab
output=./Test/output.1

python $basedir/bert/runbert/pretrain.py -i gbk \
    -c $train_data \
    -t $test_data \
    -v $vocab \
    -o $output \
    --log_freq 100 --lr 0.01 --test_freq 0 --save_freq 10000 --cuda_devices 4 5 \
    -d 0.1 --adam_weight_decay 0.0 \
    -s 42 -l 6 -a 16 -hs 256 \
    --batch_size 128 --num_workers 2 \
    --mask_loss_weight 1.0 \
    --on_memory 
