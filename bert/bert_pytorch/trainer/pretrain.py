import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..model import BERTLM, BERT
from .optim_schedule import ScheduledOptim

import tqdm
import os


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None, output_path: str="",
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 mask_loss_weight: float = 0.05,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, save_freq: int = 100,
                 test_freq: int = 0, reload_model_path: str=""):
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
        # cuda_condition = torch.cuda.is_available() and with_cuda
        # self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        if with_cuda and torch.cuda.is_available():
            if not cuda_devices:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda:%i" %cuda_devices[0])
        else:
            self.device = torch.device("cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        self.mask_loss_weight = mask_loss_weight
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size)
        self.model = self.model.to(self.device)
        self.cuda_devices = 1
        if reload_model_path:
            print("Loading Model from %s" % reload_model_path)
            self.model.load_state_dict(torch.load(reload_model_path))

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1 and cuda_devices and len(cuda_devices) > 1:
           print("Using %d GPUS for BERT" % len(cuda_devices))
           self.cuda_devices = len(cuda_devices)
           self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token

        self.log_freq = log_freq
        self.save_freq = save_freq
        self.test_freq = test_freq
        self.output_path = output_path

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def itest(self, data_loader):
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="Test",
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        avg_mask_loss = 0.0
        avg_next_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            mask_loss, next_loss, correct = self.model.forward(data["bert_input"], data["segment_label"], data["bert_label"], data["is_next"])
            mask_loss = mask_loss.mean()
            next_loss = next_loss.mean()
            correct = correct.sum()
            mask_loss = mask_loss * self.mask_loss_weight
            loss = next_loss + mask_loss

            avg_loss += loss.item()
            avg_mask_loss += mask_loss.item();
            avg_next_loss += next_loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

        print("ITEST, avg_loss=%.4f" % (avg_loss / len(data_iter)), \
              "len(data_iter)=", len(data_iter),
              "avg_mask_loss=%.4f" %(avg_mask_loss / len(data_iter)),
              "avg_next_loss=%.4f" %(avg_next_loss / len(data_iter)),
              "total_acc=%.4f" %(total_correct * 100.0 / total_element))


    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        avg_mask_loss = 0.0
        avg_next_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            mask_loss, next_loss, correct = self.model.forward(data["bert_input"], data["segment_label"], data["bert_label"], data["is_next"])
            mask_loss = mask_loss.mean()
            next_loss = next_loss.mean()
            correct = correct.sum()
            mask_loss = mask_loss * 0.05
            loss = next_loss + mask_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            avg_loss += loss.item()
            avg_mask_loss += mask_loss.item();
            avg_next_loss += next_loss.item()
            total_correct += correct.item()
            total_element += data["is_next"].nelement()
            if train:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": "%.4f" %(avg_loss / (i + 1)),
                    "avg_mask_loss": "%.4f" %(avg_mask_loss / (i+1)),
                    "avg_next_loss": "%.4f" %(avg_next_loss / (i+1)),
                   "avg_acc": "%.4f" %(total_correct / total_element * 100),
                    "loss": "%.4f" %(loss.item())
                }

                if i != 0 and i % self.log_freq == 0:
                    data_iter.write(str(post_fix))
                if i != 0 and i % self.save_freq == 0:
                    self.save(epoch, self.output_path, str(i))
                if self.test_freq and i != 0 and i % self.test_freq == 0:
                    self.itest(self.test_data)

        print("EP%d_%s, avg_loss=%.4f" % (epoch, str_code, avg_loss / len(data_iter)), \
              "len(data_iter)=", len(data_iter),
              "avg_mask_loss=%.4f" %(avg_mask_loss / len(data_iter)),
              "avg_next_loss=%.4f" %(avg_next_loss / len(data_iter)),
              "total_acc=%.4f" %(total_correct * 100.0 / total_element))
        self.save(epoch, self.output_path)

    def save(self, epoch, file_path, msg="end"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        dname = os.path.dirname(self.output_path)
        if not os.path.exists(dname):
            os.mkdir(dname)

        output_path = file_path + ".bertlm.epoch.%d.%s" % (epoch, msg)
        if self.cuda_devices > 1:
            torch.save(self.model.module.state_dict(), output_path)
        else:
            torch.save(self.model.state_dict(), output_path)


        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
