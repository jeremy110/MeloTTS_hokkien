# from . import chinese, chinese_mix, korean, french, spanish #, japanese, english
from . import cleaned_text_to_sequence
import copy
from .tai import tai

# language_module_map = {"ZH": chinese, 'ZH_MIX_EN': chinese_mix, 'KR': korean,
#                     'FR': french, 'SP': spanish, 'ES': spanish, 'TAI': tai} #, "JP": japanese, "EN": english

language_module_map = {'TAI': tai}

def clean_text(text, text_num, language, pad_start_end): 
    '''
        g2p輸入要數字調，所以這裡輸入改成以數字調輸入
    '''
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    norm_text_num = language_module.text_normalize(text_num)
    phones, tones, word2ph = language_module.g2p(norm_text_num, pad_start_end)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language, text_num, device=None):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    norm_text_num = language_module.text_normalize(text_num)
    # print(f'norm_text: {norm_text_num}')
    phones, tones, word2ph = language_module.g2p(norm_text_num) # 原先是norm_text，但台語是給台文數字調
    # print(f'phones: {phones}')
    word2ph_bak = copy.deepcopy(word2ph)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    bert = language_module.get_bert_feature(norm_text, word2ph, device=device)
    
    return norm_text, norm_text_num, phones, tones, word2ph_bak, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass