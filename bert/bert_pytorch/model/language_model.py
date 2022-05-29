import torch.nn as nn

from .bert import BERT
import torch


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)
        self.mask_criterion = nn.NLLLoss(ignore_index=0)
        self.next_criterion = nn.NLLLoss()


    def forward(self, x, segment_label, mask_label, target_label):
        x = self.bert(x, segment_label)
        next_sent_output = self.next_sentence(x)
        mask_lm_output = self.mask_lm(x)
        next_loss = self.next_criterion(next_sent_output, target_label)
        mask_loss = self.mask_criterion(mask_lm_output.transpose(1, 2), mask_label)
        correct = next_sent_output.argmax(dim=-1).eq(target_label).sum()
        return mask_loss.view(1), next_loss.view(1), correct.view(1)


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class DownstreamModel(nn.Module):

    def __init__(self, bert: BERT, classes_n):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.bert = bert
        self.linear = nn.Linear(self.bert.hidden, classes_n)
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, x, segment_info):
        x = self.bert(x, segment_info)
        return self.softmax(self.linear(x[:, 0]))
    
