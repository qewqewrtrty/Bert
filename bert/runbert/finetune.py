import argparse
import tqdm
from bert_pytorch.model import BERT
from bert_pytorch.trainer import BERTDownstreamTrainer
from bert_pytorch.dataset import DownstreamDataset, WordVocab
from torch.utils.data import DataLoader
import torch
import sys


def finetune():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")
    parser.add_argument("-p", "--pretrained_path", required=True, type=str, help="pretrained model path")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-d", "--dropout", type=float, default=0.1, help="dropout rate")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-i", "--encoding", type=str, default='gbk', help="encoding")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--label_separator", type=int, default=1, help=">=x is positive")
    parser.add_argument("--classes_n", type=int, default=2, help="classes_n")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--test_freq", type=int, default=10, help="printing test every n iter: setting n")
    parser.add_argument("--save_freq", type=int, default=10, help="saving model every n iter: setting n")
    parser.add_argument("--shuffle", type=int, default=0, help="whether shuffle data")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", dest="on_memory",action='store_true', help="Loading on memory: true or false")
    parser.add_argument("--no-on_memory", dest="on_memory",action='store_false', help="Loading on memory: true or false")


    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = DownstreamDataset(args.train_dataset, vocab, seq_len=args.seq_len, encoding=args.encoding,
                                corpus_lines=args.corpus_lines, on_memory=args.on_memory)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = DownstreamDataset(args.test_dataset, vocab, seq_len=args.seq_len,
                                     encoding=args.encoding, on_memory=args.on_memory, label_separator=args.label_separator, shuffle=args.shuffle) \
        if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) \
        if test_dataset is not None else None


    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, dropout=args.dropout)
    if args.pretrained_path:
        ps_dict = torch.load(args.pretrained_path)
        bs_dict = bert.state_dict()
        for k in bs_dict.keys():
            k1 = "bert." + k
            if k1 in ps_dict:
                bs_dict[k] = ps_dict[k1]
        bert.load_state_dict(bs_dict)
        print("Creating BERTDownstream Trainer")
        

    trainer = BERTDownstreamTrainer(bert, len(vocab), classes_n=args.classes_n, pretrained_model_path="",
                                    train_dataloader=train_data_loader, test_dataloader=test_data_loader, output_path=args.output_path,
                                    lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                    with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, test_freq=args.test_freq,
                                    save_freq=args.save_freq)

    print("Training Start")
    # ofnm = "./pred.txt"
    # trainer.predict(test_data_loader, ofnm)
    # sys.exit(0)
    for epoch in range(args.epochs):
        trainer.train(epoch)

        if test_data_loader is not None:
            trainer.test(epoch)

if __name__ == "__main__":
    finetune()
