import pickle
import os
import re


from .cleaner import tai_cleaners
from .taiwaniese_tempv3 import TLPAnumTone_to_ipa3
from .tokenizer import tokenizer




def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word

def text_normalize(text):
    text = tai_cleaners(text)
    return text

# model_id = 'dbmdz/bert-base-french-europeana-cased'
# tokenizer = AutoTokenizer.from_pretrained(model_id)

def g2p(text, pad_start_end=True, tokenized=None):
    if tokenized is None:
        t = tokenizer()
        tokenized = t.tokenize(text, have_punc = True)
    
    ph_groups = []
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            ph_groups[-1].append(t.replace("#", ""))
    
    phones = []
    tones = []
    word2ph = []
    
    for group in ph_groups:
        w = "".join(group)
        phone_len = 0
        word_len = len(group)
        
        result = re.findall(r'[a-zA-Z]+|\d+', w)
        if len(result) == 1:
            ww = result[0]
            num = 0
            # test_tone.extend([num] * len(ww))
        elif len(result) == 2:
            ww = result[0]
            num = int(result[1])
            # test_tone.extend([num] * len(ww))
        else: # 遇到標點符號
            ww = w
            num = 0
            # test_tone.extend([num] * len(ww))
        # print(f'group: {group} {num}')
        if ww == '[UNK]':
            phone_list = ['UNK']
        else:
            #phone_list = list(filter(lambda p: p != " ", fr_to_ipa.fr2ipa(w)))
            phone_list = list(filter(lambda p:p != " ", TLPAnumTone_to_ipa3(ww)))
        
        for ph in phone_list:
            phones.append(ph)
            tones.append(num)
            phone_len += 1
        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa
        # print(f'ipa: {"".join(phone_list)}')
        # print(phone_list, aaa)
        # print('=' * 10)

    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph

def get_bert_feature(text, word2ph, device=None):
    from .tai_bert import get_bert_feature
    # import tai_bert
    return get_bert_feature(text, word2ph, device = device)
    # return tai_bert.get_bert_feature(text, word2ph, device=device)

if __name__ == "__main__":
    tai = '今天天氣很好'
    tai_num_text = 'kin1-a2-jit8 thinn1-khi3 tsiok4 ho2--e5'

    tai = "初一早、初二早，初三睏甲飽。"
    tai_num_text = 'tshe1-it4 tsa2, tshe1-ji7 tsa2, tshe1-sann1 khun3 kah4 pa2.'
    tai_num_text = text_normalize(tai_num_text)
    
    tai = "隔壁親家，禮數原在。"
    tai_num_text = 'keh4-piah4 tshin1-ke1,le2-soo3 guan5-tsai7.'
    tai_num_text = text_normalize(tai_num_text)


    phones, tones, word2ph = g2p(tai_num_text)
    print(len(tai), tai)
    print(len(phones), phones)
    print(len(tones), tones)
    print(len(word2ph), word2ph)
