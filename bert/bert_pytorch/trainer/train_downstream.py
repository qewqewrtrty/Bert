import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from ..model import BERTLM, BERT, DownstreamModel
from .optim_schedule import ScheduledOptim

import tqdm
import os



class BERTDownstreamTrainer:
    """

    """

    def __init__(self, bert: BERT, vocab_size: int, classes_n: int, 
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None, output_path: str="", pretrained_model_path: str="",
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000, hidden=256,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, save_freq: int = 100, test_freq: int = 0):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        if with_cuda and torch.cuda.is_available():
            if not cuda_devices:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda:%i" %cuda_devices[0])
        else:
            self.device = torch.device("cpu")

        self.bert = bert
        self.model = DownstreamModel(bert, classes_n)
        if pretrained_model_path:
            ps_dict = torch.load(pretrained_model_path)
            self.model.load(ps_dict)

        self.model.to(self.device)
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.criterion = nn.NLLLoss()
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.test_freq = test_freq
        self.output_path = output_path

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data, train=True)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def itest(self, data_loader):
        data_iter = tqdm.tqdm(enumerate(data_loader),
                             desc="ITEST",
                             total=len(data_loader),
                             bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        all_labels = []
        all_preds = []
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            output = self.model.forward(data["bert_input"], data["segment_label"])
            loss = self.criterion(output, data["is_next"])
            avg_loss += loss.item()
            correct = output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            total_correct += correct
            total_element += data["is_next"].nelement()
            all_labels += data["is_next"].cpu().numpy().tolist()
            all_preds += output[:,1].data.cpu().numpy().tolist()
        auc = roc_auc_score(all_labels, all_preds)
        print("EPIEST, avg_loss=%.4f" % (avg_loss / len(data_iter)), \
              "len(data_iter)=", len(data_iter),
              "avg_loss=%.4f" %(avg_loss / len(data_iter)),
              "auc=%.4f" %(auc),
              "total_acc=%.4f" %(total_correct * 100.0 / total_element))

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                             desc="EP_%s:%d" % (str_code, epoch),
                             total=len(data_loader),
                             bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        all_labels = []
        all_preds = []
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            output = self.model.forward(data["bert_input"], data["segment_label"])
            loss = self.criterion(output, data["is_next"])
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            avg_loss += loss.item()
            correct = output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            total_correct += correct
            total_element += data["is_next"].nelement()
            # calculate auc
            all_labels += data["is_next"].cpu().numpy().tolist()
            all_preds += output[:,1].data.cpu().numpy().tolist()
            auc = roc_auc_score(all_labels, all_preds)
            if train:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": "%.4f" %(avg_loss / (i + 1)),
                    "avg_acc": "%.4f" %(total_correct / total_element * 100),
                    "auc": "%.4f" % auc,
                    "loss": "%.4f" %(loss.item())
                }

                if i != 0 and i % self.log_freq == 0:
                    data_iter.write(str(post_fix))
                if i != 0 and i % self.save_freq == 0:
                    self.save(epoch, self.output_path, str(i))
                if self.test_freq and i != 0 and i % self.test_freq == 0:
                    self.itest(self.test_data)

        auc = roc_auc_score(all_labels, all_preds)
        print("EP%d_%s, avg_loss=%.4f" % (epoch, str_code, avg_loss / len(data_iter)), \
              "len(data_iter)=", len(data_iter),
              "avg_loss=%.4f" %(avg_loss / len(data_iter)),
              "auc=%.4f" %(auc),
              "total_acc=%.4f" %(total_correct * 100.0 / total_element))

        self.save(epoch, self.output_path)

        
    def save(self, epoch, file_path, msg="end"):
        dname = os.path.dirname(self.output_path)
        if not os.path.exists(dname):
            os.mkdir(dname)

        output_path = file_path + ".downstream.epoch.%d.%s" % (epoch, msg)
        torch.save(self.model.state_dict(), output_path)

    def simple_test(self, data_loader):
        data_iter = tqdm.tqdm(enumerate(data_loader),
                             desc="PREDICT",
                             total=len(data_loader),
                             bar_format="{l_bar}{r_bar}")

        total_correct = 0
        total_element = 0
        all_labels = []
        all_preds = []
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            output = self.model.forward(data["bert_input"], data["segment_label"])
            correct = output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            total_correct += correct
            total_element += data["is_next"].nelement()
            all_labels += data["is_next"].cpu().numpy().tolist()
            all_preds += output[:,1].data.cpu().numpy().tolist()
        auc = roc_auc_score(all_labels, all_preds)
        post_fix = {
            "avg_acc": "%.4f" %(total_correct / total_element * 100),
            "auc": "%.4f" % auc,
        }
        print(post_fix)

class BERTDownstreamTester:
    def __init__(self, bert: BERT, vocab_size: int, classes_n: int, 
                 test_dataloader: DataLoader = None, pretrained_model_path: str="",
                 hidden=256,
                 with_cuda: bool = True, cuda_devices=None):
        if with_cuda and torch.cuda.is_available():
            if not cuda_devices:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda:%i" %cuda_devices[0])
        else:
            self.device = torch.device("cpu")

        self.bert = bert
        self.model = DownstreamModel(bert, classes_n)
        if pretrained_model_path:
            ps_dict = torch.load(pretrained_model_path)
            self.model.load_state_dict(ps_dict)
            # self.model = torch.load(pretrained_model_path)

        self.model.to(self.device)
        self.test_data = test_dataloader
        self.criterion = nn.NLLLoss()
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def test_auc(self, data_loader):
        data_iter = tqdm.tqdm(enumerate(data_loader),
                             desc="ITEST",
                             total=len(data_loader),
                             bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        all_labels = []
        all_preds = []
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            output = self.model.forward(data["bert_input"], data["segment_label"])
            loss = self.criterion(output, data["is_next"])
            avg_loss += loss.item()
            correct = output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            total_correct += correct
            total_element += data["is_next"].nelement()
            all_labels += data["is_next"].cpu().numpy().tolist()
            all_preds += output[:,1].data.cpu().numpy().tolist()
        auc = roc_auc_score(all_labels, all_preds)
        print("EPIEST, avg_loss=%.4f" % (avg_loss / len(data_iter)), \
              "len(data_iter)=", len(data_iter),
              "avg_loss=%.4f" %(avg_loss / len(data_iter)),
              "auc=%.4f" %(auc),
              "total_acc=%.4f" %(total_correct * 100.0 / total_element))
