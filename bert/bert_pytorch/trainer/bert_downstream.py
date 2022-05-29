import torch.nn as nn
import torch


class BERTDownstream(nn.Module):
    """
    BERT FineTune
    """

    def __init__(self, pretrained_path, classes_n=2):
        """
        :param get the 1st column [cls] hidden state representation from input
        """
        super().__init__()
        self.bert = torch.load(pretrained_path)
        self.linear = nn.Linear(self.bert.hidden, classes_n)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, segment_label):
        poutput = self.bert(x, segment_label)
        return self.softmax(self.linear(poutput[:,0]))
