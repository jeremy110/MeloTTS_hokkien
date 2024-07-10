import torch
from torch import nn
from transformers import BertTokenizer, BertForMaskedLM
import sys

model_id = "/ws/code/MeloTTS/melo/text/tai/en_zh_tai_hak.txt"
tokenizer = BertTokenizer.from_pretrained(model_id)
model = None


class custom_Bert(nn.Module):

    def __init__(self,):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained('hfl/chinese-roberta-wwm-ext-large', cache_dir = '/ws/code/bert_hak_tai')

    def forward(self, input_ids, token_type_ids, attention_mask, output_hidden_states=True):
        return self.bert(input_ids = input_ids, labels=None, attention_mask = attention_mask, output_hidden_states=output_hidden_states, token_type_ids = token_type_ids)


def get_bert_feature(text, word2ph, device=None):
    global model
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if model is None:
        model = custom_Bert()
        model.load_state_dict(torch.load('/ws/code/bert_hak_tai/bert_wwm_hak_tai.bin'), strict=False)
        model.eval()
        model.to(device)

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # print(res.size())
    # print(f'\n in tai_bert word2ph: {word2ph}')
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T

if __name__ == "__main__":
    get_bert_feature('今天天氣真好', [0])