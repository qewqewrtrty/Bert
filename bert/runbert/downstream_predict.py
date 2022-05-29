import argparse
import tqdm
from bert_pytorch.model import BERT
from bert_pytorch.trainer import BERTDownstreamTester
from bert_pytorch.dataset import DownstreamDataset, WordVocab
from torch.utils.data import DataLoader
import torch
import sys
import numpy as np

def predict(data_loader, ofnm, model, device=0):
    data_iter = tqdm.tqdm(enumerate(data_loader),
                         desc="PREDICT",
                         total=len(data_loader),
                         bar_format="{l_bar}{r_bar}")

    total_correct = 0
    total_element = 0
    all_labels = []
    all_preds = []
    with open(ofnm, 'w') as ofp:
        for i, data in data_iter:
            #data = {key: value.to(device) for key, value in data.items()}
            data = {key: value for key, value in data.items()}
            output = model.forward(data["bert_input"], data["segment_label"])
            correct = output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            total_correct += correct
            total_element += data["is_next"].nelement()
            all_labels += data["is_next"].cpu().numpy().tolist()
            all_preds += output[:,1].data.cpu().numpy().tolist()
            data["pred"] = output
            for bert_input, segment_label, is_next, pred in zip(
                    data["bert_input"].cpu().detach().numpy().tolist(),
                    data["segment_label"].cpu().detach().numpy().tolist(),
                    data["is_next"].cpu().detach().numpy().tolist(),
                    data["pred"].exp().cpu().detach().numpy().tolist()
            ):
                # ostr = "%s\t%s\t%s\t%s\n" %(bert_input, segment_label, is_next, pred)
                ostr = "%s\n" %(pred[1])
                ofp.write(ostr)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--test_dataset",required=True, type=str, default=None, help="test set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-o", "--output_path", type=str, help="ex)output/bert.model")
    parser.add_argument("-p", "--pretrained_path", default="", type=str, help="pretrained model path")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-d", "--dropout", type=float, default=0.1, help="dropout rate")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-i", "--encoding", type=str, default='gbk', help="encoding")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--classes_n", type=int, default=2, help="classes_n")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", dest="on_memory",action='store_true', help="Loading on memory: true or false")
    parser.add_argument("--no-on_memory", dest="on_memory",action='store_false', help="Loading on memory: true or false")
    parser.add_argument("--label_separator", type=int, default=1, help=">=x is positive")

    args = parser.parse_args()

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, dropout=args.dropout)
    test_dataset = DownstreamDataset(args.test_dataset, vocab, seq_len=args.seq_len, encoding=args.encoding, on_memory=args.on_memory,
                                     label_separator=args.label_separator)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    tester = BERTDownstreamTester(bert, len(vocab), classes_n=args.classes_n,
                                  pretrained_model_path=args.pretrained_path,
                                  test_dataloader=test_data_loader,
                                  with_cuda=args.with_cuda, cuda_devices=args.cuda_devices)

    #tester.test_auc(test_data_loader)
    predict(test_data_loader, args.output_path, tester.model, device=args.cuda_devices[0])

if __name__ == "__main__":
    main()
