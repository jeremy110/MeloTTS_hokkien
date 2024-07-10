import os
import random
import torch
import torch.utils.data
from tqdm import tqdm
from loguru import logger
import commons
from mel_processing import spectrogram_torch, mel_spectrogram_torch
from utils import load_filepaths_and_text
from utils import load_wav_to_torch_librosa as load_wav_to_torch
from text import cleaned_text_to_sequence, get_bert
import numpy as np
import re

"""Multi speaker version"""

def insert_SP(phone, tone, language, word2ph, res_SP, sp_symbol = 217):
    
    new_phone = []
    new_tone = []
    new_language = []
    prev_idx = 0
    success_flag = False
    try:
        for sp in res_SP:
            idx = sum(word2ph[: sp])
            new_phone.extend(phone[prev_idx: idx] + [sp_symbol])
            new_tone.extend(tone[prev_idx: idx] + [0])
            new_language.extend(language[prev_idx: idx] + [0])
            prev_idx = idx

        new_phone.extend(phone[prev_idx: ])
        new_tone.extend(tone[prev_idx: ])
        new_language.extend(language[prev_idx: ])
        success_flag = True
        return new_phone, new_tone, new_language, success_flag
    except:
        return phone, tone, language, success_flag


def insert_zero_vector(word2phone, res_SP, bert_emb):

    zero_vec = torch.zeros(1024, 1)
    final_bert_emb = []
    pred_idx = 0
    for sp in res_SP:
        idx = sum(word2phone[: sp])
        final_bert_emb.append(bert_emb[:, pred_idx: idx])
        final_bert_emb.append(zero_vec)
        pred_idx = idx

    final_bert_emb.append(bert_emb[:, pred_idx: ])
    final_bert_emb = torch.cat(final_bert_emb, dim = -1)
        
    # print(f'final_bert_emb {final_bert_emb.size(-1)}')
    return final_bert_emb


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.spk_map = hparams.spk2id
        self.hparams = hparams
        self.disable_bert = getattr(hparams, "disable_bert", False)

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 300)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()


    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        skipped = 0
        logger.info("Init dataset...")
        for item in tqdm(
            self.audiopaths_sid_text
        ):
            try: # New: add text_num for sp symbol
                _id, spk, language, text, phones, tone, word2ph, text_num = item 
                print(item)
            except:
                print(item)
                raise
            audiopath = f"{_id}"
            if self.min_text_len <= len(phones) and len(phones) <= self.max_text_len:
                phones = phones.split(" ")
                tone = [int(i) for i in tone.split(" ")]
                word2ph = [int(i) for i in word2ph.split(" ")]
                audiopaths_sid_text_new.append(
                    [audiopath, spk, language, text, phones, tone, word2ph, text_num]
                )
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            else:
                skipped += 1
        logger.info(f'min: {min(lengths)}; max: {max(lengths)}' )
        logger.info(
            "skipped: "
            + str(skipped)
            + ", total: "
            + str(len(self.audiopaths_sid_text))
        )
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, language, text, phones, tone, word2ph, text_num = audiopath_sid_text

        bert, ja_bert, phones, tone, language = self.get_text(
            text, word2ph, phones, tone, language, audiopath, text_num
        )

        spec, wav = self.get_audio(audiopath)
        sid = int(getattr(self.spk_map, sid, '0'))
        sid = torch.LongTensor([sid])
        return (phones, spec, wav, sid, tone, language, bert, ja_bert)

    def get_audio(self, filename):
        audio_norm, sampling_rate = load_wav_to_torch(filename, self.sampling_rate)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    filename, sampling_rate, self.sampling_rate
                )
            )
        # NOTE: normalize has been achieved by torchaudio
        # audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        try:
            spec = torch.load(spec_filename)
            assert False
        except:
            if self.use_mel_spec_posterior:
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text, word2ph, phone, tone, language_str, wav_path, text_num):
        phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
        if self.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1

            # New: 如果是字跟字中間的話將 0 取代成 SP(217)
            res_SP = self.ckip(text_num)
            # phone = self.add_SP(phone, word2ph, res_SP)
            phone, tone, language, success_flag = insert_SP(phone, tone, language, word2ph, res_SP)

        bert_path = wav_path.replace(".wav", ".bert.pt")
        try:
            bert = torch.load(bert_path)
            # New
            if success_flag:
                bert = insert_zero_vector(word2ph, res_SP, bert)

            assert bert.shape[-1] == len(phone)
        except Exception as e:
            print(e, wav_path, bert_path, bert.shape, len(phone))
            bert = get_bert(text, word2ph, language_str)
            torch.save(bert, bert_path)
            assert bert.shape[-1] == len(phone), phone

        
        if self.disable_bert:
            bert = torch.zeros(1024, len(phone))
            ja_bert = torch.zeros(768, len(phone))
        else:
            if language_str in ["ZH"]:
                bert = bert
                ja_bert = torch.zeros(768, len(phone))
            elif language_str in ["JP", "EN", "ZH_MIX_EN", "KR", 'SP', 'ES', 'FR', 'DE', 'RU']:
                ja_bert = bert
                bert = torch.zeros(1024, len(phone))
            elif language_str in ["TAI"]:
                bert = bert
                ja_bert = torch.zeros(768, len(phone))
            else:
                raise
                bert = torch.zeros(1024, len(phone))
                ja_bert = torch.zeros(768, len(phone))
        assert bert.shape[-1] == len(phone)
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tone)
        language = torch.LongTensor(language)
        return bert, ja_bert, phone, tone, language

    def ckip(self, text_num):
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

    def add_SP(self, phone, word2phone, res_SP, sp_symbol = 217):
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
            
        # print(phone)
        

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        language_padded = torch.LongTensor(len(batch), max_text_len)
        bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        ja_bert_padded = torch.FloatTensor(len(batch), 768, max_text_len)

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        tone_padded.zero_()
        language_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        bert_padded.zero_()
        ja_bert_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            tone = row[4]
            tone_padded[i, : tone.size(0)] = tone

            language = row[5]
            language_padded[i, : language.size(0)] = language

            bert = row[6]
            bert_padded[i, :, : bert.size(1)] = bert

            ja_bert = row[7]
            ja_bert_padded[i, :, : ja_bert.size(1)] = ja_bert

        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            sid,
            tone_padded,
            language_padded,
            bert_padded,
            ja_bert_padded,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
        print('buckets:', self.num_samples_per_bucket)

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        try:
            for i in range(len(buckets) - 1, 0, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)
            assert all(len(bucket) > 0 for bucket in buckets)
        # When one bucket is not traversed
        except Exception as e:
            print("Bucket warning ", e)
            for i in range(len(buckets) - 1, -1, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            if len_bucket == 0:
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
