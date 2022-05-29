basedir=./
export PYTHONPATH=./bert
python $basedir/bert/runbert/downstream_predict.py -i gbk \
    -t ./Test/data/finetune/sts_b.test \
    -v ./Test/data/pretrain/vocab \
    -o ./Test/data/finetune/sts_b.test.predict.score \
    -b 128 \
    -s 42 -l 6 -a 16 -hs 256 \
    --cuda_devices 8 \
    --num_workers 1 \
    --label_separator 2 \
    -p ./Test/output.finetune.1.downstream.epoch.0.30

