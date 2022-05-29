basedir=./
corpus=./Test/data/pretrain/corpus
vocab=./Test/data/pretrain/vocab
export PYTHONPATH=./bert
python $basedir/bert/runbert/build_vocab.py -c $corpus -o $vocab -s 80000 -e gbk
