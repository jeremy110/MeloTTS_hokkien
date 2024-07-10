import os
import glob
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch
import torchaudio
import librosa
from melo.text import cleaned_text_to_sequence, get_bert
from melo.text.cleaner import clean_text
from melo import commons
import re

MATPLOTLIB_FLAG = False

logger = logging.getLogger(__name__)

def ckip(text_num):
    text_num = text_num.replace('--', '-').replace('？', '').replace('.', '。')

    text_num_arr = re.split(r'\s+|-|(:)|(,)|(。)|(？)|(!)', text_num)
    text_num_arr = [item for item in text_num_arr if item]

    w = re.split(r'\s+|(:)|(,)|(。)|(？)|(!)', text_num)
    w = [item for item in w if item]

    a_idx = 0
    prev_is_punct = False
    punct_arr = [',', ':', '。', '？', '!']
    res_SP = []

    for idx, val in enumerate(w):
        char_arr = val.split('-')
        L = len(char_arr)
        if L == 1:
            if val in punct_arr:
                prev_is_punct = True
            else:
                if prev_is_punct == False and idx != 0:
                    res_SP.append(a_idx + 1)
                prev_is_punct = False
            a_idx += 1
        else:
            if prev_is_punct == False and idx != 0:
                res_SP.append(a_idx + 1)
            a_idx += L
    
    return res_SP

def add_SP(phone, word2phone, res_SP, sp_symbol = 217):
    # phone = [0, 0, 0, 51, 0, 87, 0, 51, 0, 124, 0, 119, 0, 211, 0, 89, 0, 76, 0, 87, 0, 51, 0, 46, 0, 93, 0, 19, 0, 51, 0, 119, 0, 70, 0, 19, 0, 119, 0, 37, 0, 82, 0, 198, 0, 200, 0, 201, 0, 198, 0, 51, 0, 68, 0, 211, 0, 0, 0]
    # phone = [0, 0, 0, 51, 0, 87, 0, 51, 0, 124, 0, 119, 0, 211, 0, 89, 0, 76, 1, 87, 0, 51, 1, 46, 0, 93, 0, 19, 1, 51, 0, 119, 0, 70, 0, 19, 0, 119, 1, 37, 1, 82, 0, 198, 0, 200, 0, 201, 0, 198, 0, 51, 0, 68, 0, 211, 0, 0, 0]
    # word2phone = [3, 2, 8, 2, 4, 4, 6, 4, 6, 2, 6, 8, 2, 2]
    # print('\nphone, res_sp', phone, res_SP)
    new_phone = phone.copy()
    try:
        for sp in res_SP:
            idx = sum(word2phone[: sp])
            new_phone[idx - 1] = sp_symbol
        return new_phone
    except:
        return phone
    
def insert_SP(phone, tone, language, word2ph, res_SP, sp_symbol = 217):
    
    new_phone = []
    new_tone = []
    new_language = []
    prev_idx = 0

    try:
        for sp in res_SP:
            idx = sum(word2ph[: sp])
            new_phone.extend(phone[prev_idx: idx] + [sp_symbol])
            new_tone.extend(tone[prev_idx: idx] + [0])
            new_language.extend(language[prev_idx: idx] + [0])
            # print(f'new_phone {new_phone}')
            prev_idx = idx
        new_phone.extend(phone[prev_idx: ])
        new_tone.extend(tone[prev_idx: ])
        new_language.extend(language[prev_idx: ])
        # print(f'final new_phone {new_phone}')
        return new_phone, new_tone, new_language
    except:
        return phone, tone, language


def insert_zero_vector(word2phone, res_SP, bert_emb):

    zero_vec = torch.zeros(1024, 1)
    final_bert_emb = []
    pred_idx = 0
    for sp in res_SP:
        idx = sum(word2phone[: sp])
        final_bert_emb.append(bert_emb[:, pred_idx: idx])
        final_bert_emb.append(zero_vec)
        pred_idx = idx

        # step = word2phone[sp - 1] // 2
        # print(f'inset idx: {idx} {step}')
        # zero_vec = torch.zeros(1024, step)
        # bert_emb[:, idx - step - 1: idx - 1] = zero_vec
        # bert_emb[:, idx - 1] = zero_vec
    final_bert_emb.append(bert_emb[:, pred_idx: ])
    final_bert_emb = torch.cat(final_bert_emb, dim = -1)
        
    print(f'final_bert_emb {final_bert_emb.size(-1)}')
    return final_bert_emb

def get_text_for_tts_infer(text, text_num, language_str, hps, device, symbol_to_id=None):
    '''
        text: 台語漢字 --> for bert extract embedding 
        text_norm:  台文數字調 --> for g2p(to ipa)
    '''
    norm_text, phone, tone, word2ph = clean_text(text, text_num, language_str, pad_start_end = True)
    print('\n====clean_text=====')
    print('norm_text', len(norm_text), norm_text)
    print('phone', len(phone), phone)
    print('tone', len(tone), tone)
    print('word2ph', len(word2ph), sum(word2ph), word2ph)

    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, symbol_to_id)
    print('\n========cleaned_text_to_sequence=========')
    print('text_num', text_num)
    print('phone', phone)
    print('tone', tone)
    # hps.data.add_blank = False
    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

        # New: 如果是字跟字中間的話將 0 取代成 SP(217)
        res_SP = ckip(text_num)
        # phone = add_SP(phone, word2ph, res_SP)
        phone, tone, language = insert_SP(phone, tone, language, word2ph, res_SP)
    print(f'add_blank phone:{len(phone)} {phone}')
    print(f'add_blank tone:{len(tone)} {tone}')
    print(f'add_blank language:{len(language)} {language}')
    print(f'add_blank word2ph: {word2ph}')
    print(f'res_SP: {res_SP}')
    
    if getattr(hps.data, "disable_bert", False):
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))
    else:
        bert = get_bert(norm_text, word2ph, language_str, device)

        # New: we record the index of SP and 
        # AP and insert the all-zero vector in the corresponding position
        # of the sequence.
        bert = insert_zero_vector(word2ph, res_SP, bert)
    
        assert bert.shape[-1] == len(phone), phone
        
        del word2ph
        if language_str == "ZH":
            bert = bert
            ja_bert = torch.zeros(768, len(phone))
        elif language_str in ["JP", "EN", "ZH_MIX_EN", 'KR', 'SP', 'ES', 'FR', 'DE', 'RU']:
            ja_bert = bert
            bert = torch.zeros(1024, len(phone))
        elif language_str == "TAI":
            bert = bert
            ja_bert = torch.zeros(768, len(phone))
        else:
            raise NotImplementedError()

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language

def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict.get("iteration", 0)
    learning_rate = checkpoint_dict.get("learning_rate", 0.)
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    elif optimizer is None and not skip_optimizer:
        # else:      Disable this line if Infer and resume checkpoint,then enable the line upper
        new_opt_dict = optimizer.state_dict()
        new_opt_dict_params = new_opt_dict["param_groups"][0]["params"]
        new_opt_dict["param_groups"] = checkpoint_dict["optimizer"]["param_groups"]
        new_opt_dict["param_groups"][0]["params"] = new_opt_dict_params
        optimizer.load_state_dict(new_opt_dict)

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "emb_g" not in k
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except Exception as e:
            print(e)
            # For upgrading from the old version
            if "ja_bert_proj" in k:
                v = torch.zeros_like(v)
                logger.warn(
                    f"Seems you are using the old version of the model, the {k} is automatically set to zero for backward compatibility"
                )
            else:
                logger.error(f"{k} is not in the checkpoint")

            new_state_dict[k] = v

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    logger.info(
        "Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration)
    )

    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_wav_to_torch_new(full_path):
    audio_norm, sampling_rate = torchaudio.load(full_path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True)
    audio_norm = audio_norm.mean(dim=0)
    return audio_norm, sampling_rate

def load_wav_to_torch_librosa(full_path, sr):
    audio_norm, sampling_rate = librosa.load(full_path, sr=sr, mono=True)
    return torch.FloatTensor(audio_norm.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/base.json",
        help="JSON file for configuration",
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--port', type=int, default=10000)
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
    parser.add_argument('--pretrain_G', type=str, default=None,
                            help='pretrain model')
    parser.add_argument('--pretrain_D', type=str, default=None,
                            help='pretrain model D')
    parser.add_argument('--pretrain_dur', type=str, default=None,
                            help='pretrain model duration')

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    os.makedirs(model_dir, exist_ok=True)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    hparams.pretrain_G = args.pretrain_G
    hparams.pretrain_D = args.pretrain_D
    hparams.pretrain_dur = args.pretrain_dur
    hparams.port = args.port
    return hparams


def clean_checkpoints(path_to_models="logs/44k/", n_ckpts_to_keep=2, sort_by_time=True):
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  Path to the model directory
    n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
    sort_by_time      --  True -> chronologically delete ckpts
                          False -> lexicographically delete ckpts
    """
    import re

    ckpts_files = [
        f
        for f in os.listdir(path_to_models)
        if os.path.isfile(os.path.join(path_to_models, f))
    ]

    def name_key(_f):
        return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))

    def time_key(_f):
        return os.path.getmtime(os.path.join(path_to_models, _f))

    sort_key = time_key if sort_by_time else name_key

    def x_sorted(_x):
        return sorted(
            [f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")],
            key=sort_key,
        )

    to_del = [
        os.path.join(path_to_models, fn)
        for fn in (x_sorted("G")[:-n_ckpts_to_keep] + x_sorted("D")[:-n_ckpts_to_keep])
    ]

    def del_info(fn):
        return logger.info(f".. Free up space by deleting ckpt {fn}")

    def del_routine(x):
        return [os.remove(x), del_info(x)]

    [del_routine(fn) for fn in to_del]


def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn(
            "{} is not a git repository, therefore hash value comparison will be ignored.".format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
