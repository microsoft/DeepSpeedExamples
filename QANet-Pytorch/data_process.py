# -*- coding: utf-8 -*-
import os
import sys
import pickle
import ujson
import spacy
import random
from tqdm import tqdm
from collections import Counter
import numpy as np

def pickle_dump_large_file(obj, filepath):
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])

def pickle_load_large_file(filepath):
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj

class Data_process(object):

    def __init__(self, args):
        self.config = args
        self.NLP = spacy.blank('en')

    def process(self):
        word_counter = Counter()
        char_counter = Counter()
        train_preprocessed_data, train_premeta_data, train_eval_data = self.data_preprocess(self.config.train_data,
                                                                                            'train',word_counter,
                                                                                            char_counter)
        dev_preprocessed_data, dev_premeta_data, dev_eval_data = self.data_preprocess(self.config.dev_data, 'dev',
                                                                                word_counter, char_counter)

        word_embedding_original = self.config.glove_word_embedding
        word_embedding_size = self.config.glove_word_size
        word_embedding_dim = self.config.glove_dim

        char_embedding_pretrained = self.config.char_emb_pretrained
        char_embedding_original = self.config.glove_char_embedding if char_embedding_pretrained else None
        char_embedding_size = self.config.glove_char_size if char_embedding_pretrained else None
        char_embedding_dim = self.config.glove_dim if char_embedding_pretrained else self.config.pretrained_char_emb_dim

        word_embedding_matrix, word2idx_dict = self.embedding_preprocess(word_counter, 'word',
                                                                         emb_original_file=word_embedding_original,
                                                                         emb_size=word_embedding_size,
                                                                         emb_dim=word_embedding_dim)
        char_embedding_matrix, char2idx_dict = self.embedding_preprocess(char_counter, 'char',
                                                                         emb_original_file=char_embedding_original,
                                                                         emb_size=char_embedding_size,
                                                                         emb_dim=char_embedding_dim)
        train_processed_data, train_meta_data = self.data_finalprocess(train_preprocessed_data,
                                                                       train_premeta_data, 'train',
                                                                       word2idx_dict, char2idx_dict)
        dev_processed_data, dev_meta_data = self.data_finalprocess(dev_preprocessed_data,
                                                                    dev_premeta_data, 'dev', word2idx_dict,
                                                                   char2idx_dict)
        self.save(self.config.processed_word_embedding, word_embedding_matrix, message='word embedding')
        self.save(self.config.processed_char_embedding, char_embedding_matrix, message='char embedding')
        self.save(self.config.word_dictionary, word2idx_dict, message='word dictionary')
        self.save(self.config.char_dictionary, char2idx_dict, message='char dictionary')
        self.save(self.config.train_processed_data, train_processed_data, message='processed train data')
        self.save(self.config.dev_processed_data, dev_processed_data, message='processed dev data')
        self.save(self.config.train_meta_data, train_meta_data, message='train meta data')
        self.save(self.config.dev_meta_data, dev_meta_data, message='dev meta data')
        self.save(self.config.train_eval_data, train_eval_data, message='train eval data')
        self.save(self.config.dev_eval_data, dev_eval_data, message='dev eval data')

    def data_preprocess(self, data_path, data_type, word_counter, char_counter, debug=False, debug_lenth=1):
        print('Preprocessing {} data...'.format(data_type))
        preprocessed_data = []
        meta_data = {}
        eval_data = {}

        with open(data_path,'r') as file:
            data_dic = ujson.load(file)
            version = data_dic['version']
            meta_data['version'] = version
            meta_data['num_q'] = 0
            meta_data['num_q_answerable'] = 0
            meta_data['num_a_answerable'] = 0
            meta_data['num_q_noanswer'] = 0
            for article in tqdm(data_dic['data']):
                for paragraph in article['paragraphs']:
                    context = paragraph['context'].replace("''", '" ').replace("``", '" ')
                    tokens_context = self.tokenizer(context)
                    chars_context = [list(token) for token in tokens_context]
                    spans = self.get_spans(context, tokens_context)
                    for token in tokens_context:
                        word_counter[token] += len(paragraph['qas'])
                        for char in token:
                            char_counter[char] += len(paragraph['qas'])
                    for qa in paragraph['qas']:
                        meta_data['num_q'] += 1
                        question = qa['question'].replace("''", '" ').replace("``", '" ')
                        tokens_question = self.tokenizer(question)
                        chars_question = [list(token) for token in tokens_question]
                        for token in tokens_question:
                            word_counter[token] += 1
                            for char in token:
                                char_counter[char] += 1
                        if version == '1.1':
                            answers = qa['answers']
                            answerable = 1
                        elif version == 'v2.0' and qa['is_impossible'] is True:
                            answers = qa['plausible_answers']
                            answerable = 0
                        meta_data['num_q_answerable'] += answerable
                        if len(answers) == 0:
                            meta_data['num_q_noanswer'] += 1
                            continue
                        a_starts, a_ends = [], []
                        answer_texts = []
                        for answer in answers:
                            answer_text = answer['text']
                            answer_start = answer['answer_start']
                            answer_end = answer_start + len(answer_text)
                            answer_texts.append(answer_text)
                            answer_span = []
                            for idx, span in enumerate(spans):
                                if answer_start < span[1] and answer_end > span[0]:
                                    answer_span.append(idx)
                            a_starts.append(answer_span[0])
                            a_ends.append(answer_span[-1])
                            meta_data['num_a_answerable'] += answerable
                        preprocessed_data_single = {
                            'tokens_context': tokens_context,
                            'chars_context': chars_context,
                            'tokens_question': tokens_question,
                            'chars_question': chars_question,
                            'a_starts': a_starts,
                            'a_ends': a_ends,
                            'id': meta_data['num_q'],
                            'spans': spans,
                            'answerable': answerable,
                        }
                        preprocessed_data.append(preprocessed_data_single)
                        eval_data[str(meta_data['num_q'])] = {
                            'context': context,
                            'spans': spans,
                            'answers': answer_texts,
                            'uuid': qa['id']
                        }
                        if debug and debug_lenth <= meta_data['num_q']:
                            return preprocessed_data, eval_data
            random.shuffle(preprocessed_data)
            print('{} questions in total have been preprocessed'.format(len(preprocessed_data)))
        return preprocessed_data, meta_data, eval_data

    def embedding_preprocess(self, counter, data_type, emb_original_file=None, emb_size=None, emb_dim=None, limit=-1,
                             specials=['<PAD>', '<OOV>', '<SOS>', '<EOS>']):
        print('Proprocessing {} embedding...'.format(data_type))
        token2embedding_dict = {}
        filtered_elements = [k for k, v in counter.items() if v > limit]
        if emb_original_file is None:
            assert emb_dim is not None
            for token in filtered_elements:
                token2embedding_dict[token] = [np.random.normal(scale=0.1) for _ in range(emb_dim)]
            print('{} tokens have corresponding {} embedding vector'.format(len(filtered_elements), data_type))
        else:
            assert emb_size is not None
            assert emb_dim is not None
            with open(emb_original_file, 'r', encoding='utf-8') as file:
                for line in tqdm(file, total=emb_size):
                    vec_list = line.split()
                    token = "".join(vec_list[0:-emb_dim])
                    token_vector = list(map(float, vec_list[-emb_dim:]))
                    if token in counter and counter[token] > limit:
                        token2embedding_dict[token] = token_vector
            print('{} / {} tokens have corresponding {} embedding vector'.format(len(token2embedding_dict),
                                                                                 len(filtered_elements), data_type))

        token2idx_dict = {token : idx for idx, token in enumerate(token2embedding_dict.keys(), len(specials))}
        for i in range(len(specials)):
            token2idx_dict[specials[i]] = i
            token2embedding_dict[specials[i]] = [0. for _ in range(emb_dim)]
        idx2embedding_dict = {idx : token2embedding_dict[token] for token, idx in token2idx_dict.items()}
        embedding_matrix = [idx2embedding_dict[idx] for idx in range(len(idx2embedding_dict))]

        return embedding_matrix, token2idx_dict

    def data_finalprocess(self, preprocessed_data, premeta_data, data_type, word2idx_dict, char2idx_dict):
        print('Final processing {} data...'.format(data_type))
        final_processed_data = []
        total_singles = 0
        total_not_longer_singles = 0
        for preprocessed_data_single in tqdm(preprocessed_data):
            total_singles += 1
            if self.filter_longer_function(preprocessed_data_single):
                continue
            total_not_longer_singles += 1

            wids_context = np.ones([self.config.context_limit], dtype=np.int32) * word2idx_dict['<PAD>']
            cids_context = np.ones([self.config.context_limit, self.config.char_limit], dtype=np.int32) * \
                       char2idx_dict['<PAD>']
            wids_question = np.ones([self.config.question_limit], dtype=np.int32) * word2idx_dict['<PAD>']
            cids_question = np.ones([self.config.question_limit, self.config.char_limit], dtype=np.int32) * \
                        char2idx_dict['<PAD>']

            for idx, token in enumerate(preprocessed_data_single['tokens_context']):
                wids_context[idx] = self.word2wid(token, word2idx_dict)

            for idx, token in enumerate(preprocessed_data_single['tokens_question']):
                wids_question[idx] = self.word2wid(token, word2idx_dict)

            for idx, token in enumerate(preprocessed_data_single['chars_context']):
                for i, char in enumerate(token):
                    if i == self.config.char_limit:
                        break
                    cids_context[idx, i] = self.char2cid(char, char2idx_dict)

            for idx, token in enumerate(preprocessed_data_single['chars_question']):
                for i, char in enumerate(token):
                    if i == self.config.char_limit:
                        break
                    cids_question[idx, i] = self.char2cid(char, char2idx_dict)

            #using the last answer as the target answer
            start, end = preprocessed_data_single['a_starts'][-1], preprocessed_data_single['a_ends'][-1]


            preprocessed_data_single['wids_context'] = wids_context
            preprocessed_data_single['cids_context'] = cids_context
            preprocessed_data_single['wids_question'] = wids_question
            preprocessed_data_single['cids_question'] = cids_question
            preprocessed_data_single['a_start'] = start
            preprocessed_data_single['a_end'] = end
            #Save memory
            preprocessed_data_single['spans'] = None
            preprocessed_data_single['tokens_context'] = None
            preprocessed_data_single['chars_context'] = None
            preprocessed_data_single['tokens_question'] = None
            preprocessed_data_single['chars_question'] = None
            preprocessed_data_single['a_starts'] = None
            preprocessed_data_single['a_ends'] = None

            final_processed_data.append(preprocessed_data_single)

        print('Final processing {} / {} instances of data in total'.format(total_not_longer_singles, total_singles))
        premeta_data['num_q_filtered'] = total_not_longer_singles
        return final_processed_data, premeta_data

    def tokenizer(self, context):
        content = self.NLP(context)
        tokens = [token.text for token in content]
        return tokens

    def get_spans(self, context, tokens):
        current = 0
        spans = []
        for token in tokens:
            current = context.find(token, current)
            if current < 0:
                print('Token {} can not be found'.format(token))
                raise Exception()
            spans.append((current, current + len(token)))
            current += len(token)
        return spans

    def filter_longer_function(self,preprocessed_data_single):

        return (len(preprocessed_data_single['tokens_context']) > self.config.context_limit or
                len(preprocessed_data_single['tokens_question']) > self.config.question_limit or
                (preprocessed_data_single['a_ends'][0]-preprocessed_data_single['a_starts'][0]) > self.config.answer_limit)

    def word2wid(self, token, word2idx_dict):
        for single in (token, token.lower(), token.capitalize(), token.upper()):
            if single in word2idx_dict:
                return word2idx_dict[single]
        return word2idx_dict['<OOV>']

    def char2cid(self, char, char2idx_dict):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return char2idx_dict['<OOV>']

    def save(self,filepath, obj, message=None):
        if message is not None:
            print('Saving {} ...'.format(message))
        pickle_dump_large_file(obj, filepath)