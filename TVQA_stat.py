import glob
import json
import os
import re
from collections import Counter

from symspellpy.symspellpy import SymSpell, Verbosity


# borrow from vqaEval.py
m_contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've",
    "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't",
    "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
    "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock",
    "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've",
    "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's",
    "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "twas": "'twas",
    "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
    "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've"}
m_manual_map = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
                'four': '4', 'five': '5', 'six': '6', 'seven': '7',
                'eight': '8', 'nine': '9', 'ten': '10'}
m_articles = ['a', 'an', 'the']
m_period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
m_comma_strip = re.compile("(\d)(\,)(\d)")
m_punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+',
           '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']

def process_punct(in_text):
    out_text = in_text
    for p in m_punct:
        if (p + ' ' in in_text or ' ' + p in in_text
                or re.search(m_comma_strip, in_text) != None):
            out_text = out_text.replace(p, '')
        else:
            out_text = out_text.replace(p, ' ')
    out_text = m_period_strip.sub("", out_text, re.UNICODE)
    return out_text

def process_digit_article(in_text):
    out_text = []
    for word in in_text.lower().split():
        if word not in m_articles:
            word = m_manual_map.setdefault(word, word)
            word = m_contractions.setdefault(word, word)
            out_text.append(word)
    return ' '.join(out_text)

class Stat(object):
    def __init__(self):
        self.train_file_path = "./dataTVQA/TextVQA_0.5_train.json"
        self.valid_file_path = "./dataTVQA/TextVQA_0.5_val.json"
        self.test_file_path = "./dataTVQA/TextVQA_0.5_test.json"
        self.top_ans_path = "./dataTVQA/data.json"
        with open(self.train_file_path, "r") as train_file:
            self.train_dict = json.load(train_file)
        with open(self.valid_file_path, "r") as valid_file:
            self.valid_dict = json.load(valid_file)
        with open(self.test_file_path, "r") as test_file:
            self.test_dict = json.load(test_file)
        with open(self.top_ans_path, "r") as top_ans_file:
            self.top_ans_file = json.load(top_ans_file)["itoa"]
        self.max_edit_distance_lookup = 1
        self.max_edit_distance_dictionary = 2
        self.prefix_length = 13
        self.path_dict = "./english_frequency.txt"
        self.load_dictionary()

    def format_text(self, text, is_remove_irregular=0):
        """
        @description:
        format the input
        """
        if is_remove_irregular == 1:
            rstr = r"[\=\(\)\,\/\\\:\*\?\"\<\>\|\' ']"
            text = re.sub(rstr, "", text)
        text = text.lower()
        text = self.norm_answer(text)
        return text

    def norm_answer(self, ans):
        ans = ans.lower().replace('\n', ' ').replace('\t', ' ').strip()
        ans = process_punct(ans)
        ans = process_digit_article(ans)
        return ans

    def build_corpus(self, path_raw="./raw_data", path_corpus="./corpus.txt"):
        """
        @description:
        merge raw text files into one corpus file

        @path_raw: the dictory of the raw files, this function will recursivly find all the txt file in this dictory
        @path_corpus: output file path
        """
        corpus_file = open(path_corpus, "a+")
        for filename in glob.iglob(path_raw + "/**/*.txt", recursive=True):
            f = open(filename, "r")
            corpus_file.write(f.read() + "\n")
        print("finish corpus {} ~".format(path_corpus))

    def build_dictionary(self, is_already_build_corpus=1, path_corpus="./corpus.txt"):
        """
        @description:
        build the freqency english dictionary, by collecting the context of corpus file
        the first column is the word, and the second col is the word frequency
        """
        if is_already_build_corpus == 0:
            self.build_corpus()
        sym_spell = SymSpell(self.max_edit_distance_dictionary, self.prefix_length)

        if not sym_spell.create_dictionary(path_corpus):
            print("Corpus file not found")
            return
        print("finish load corpus {} ~".format(path_corpus))

        fre_dict = open(self.path_dict, "w")

        for key, count in sym_spell.words.items():
            fre_dict.write("{} {}\n".format(key, count))
        print("finish build frequency dictionary {} ~".format(self.path_dict))

    def load_dictionary(self, is_already_build_dict=1):
        """
        @description:
        load the frequency dictionary, and save the spelling object
        """
        if is_already_build_dict == 0:
            self.build_dictionary()
        self.sym_spell = SymSpell(self.max_edit_distance_dictionary, self.prefix_length)
        dictionary_path = os.path.join(os.path.dirname(__file__), self.path_dict)
        term_index = 0
        count_index = 1
        if not self.sym_spell.load_dictionary(dictionary_path, term_index, count_index):
            print("Dictionary file not found")
            return

    def spell_check_single_word(self, word):
        """
        @descrition:
        do the spell-checking for a single word, if there is no matched item, return the input itself
        """
        if word is "":
            return
        suggestion_verbosity = Verbosity.CLOSEST  # TOP, CLOSEST, ALL
        suggestions = self.sym_spell.lookup(
            word, suggestion_verbosity, self.max_edit_distance_lookup
        )
        if len(suggestions) == 0:
            return word
        return suggestions[0].term

    def stat_zero_ocr_token(self, is_train=1, is_valid=1, is_test=1):
        """
        @description:
        collecting the item in which there is zero length ocr token

        @is_train: for training set
        @is_valid: for validation set
        @is_test: for testing set
        """
        if is_train:
            zero_num = 0
            sum_num = 0
            print("{:=^46}".format('Training Set Zero Ocr Token Stat'))
            for context in self.train_dict["data"]:
                if len(context["ocr_tokens"]) == 0:
                    zero_num += 1
                sum_num += 1
            print("# zero ocr token: {}\n# len of train set: {}\n  rate {:.2%}\n".format(
                    zero_num, sum_num, 1.0 * zero_num / sum_num))

        if is_valid:
            zero_num = 0
            sum_num = 0
            print("{:=^46}".format('Validation Set Zero Ocr Token Stat'))
            for context in self.valid_dict["data"]:
                if len(context["ocr_tokens"]) == 0:
                    zero_num += 1
                sum_num += 1
            print("# zero ocr token: {}\n# len of validation set: {}\n  rate {:.2%}\n".format(
                    zero_num, sum_num, 1.0 * zero_num / sum_num))

        if is_test:
            zero_num = 0
            sum_num = 0
            print("{:=^46}".format('Testation Set Zero Ocr Token Stat'))
            for context in self.test_dict["data"]:
                if len(context["ocr_tokens"]) == 0:
                    zero_num += 1
                sum_num += 1
            print("# zero ocr token: {}\n# len of testing set: {}\n  rate {:.2%}\n".format(
                    zero_num, sum_num, 1.0 * zero_num / sum_num))

    def stat_corr_answer(
        self,
        is_spell_check=0,
        is_in_ocr=1,
        is_in_top_ans=1,
        is_splice=1,
        is_over_one=1,
        is_valid=1,
        is_train=1,
    ):
        """
        @ description: a general stat func
        """
        print("{:=^46}".format('General Stat Function'))
        test_set = []
        if is_train:
            test_set.append([self.train_dict, 1])
        if is_valid:
            test_set.append([self.valid_dict, 2])

        for temp in test_set:
            [_set, value] = temp
            if value == 1:
                print("TRAIN")
            elif value == 2:
                print("")
                print("-"*46)
                print("VALIDATION")

            if is_in_ocr == 1:
                print("\nocr token is exactly answer")
                corr = 0
                all_ans = 0
                for context in _set["data"]:
                    answers = [self.norm_answer(ans) for ans in context["answers"]]
                    ocr_tokens = [
                        self.format_text(token) for token in context["ocr_tokens"]
                    ]
                    if is_spell_check:
                        ocr_tokens = [
                            self.spell_check_single_word(w) for w in ocr_tokens
                        ]
                    all_ans += 1
                    for ans in answers:
                        if ans in ocr_tokens:
                            corr += 1
                            break
                print(
                    "# all answers: {}\t# answers in ocr tokens: {}".format(
                        all_ans, corr
                    )
                )
                print("  rate: {:.2%}".format(1.0 * corr / all_ans))

            if is_in_top_ans:
                print("\ntop answer file contains answer")
                corr = 0
                all_ans = 0
                for context in _set["data"]:
                    answers = [self.norm_answer(ans) for ans in context["answers"]]
                    all_ans += 1
                    for ans in answers:
                        if ans in self.top_ans_file:
                            corr += 1
                            break
                print(
                    "# all answers: {}\t# answers in ocr tokens: {}".format(
                        all_ans, corr
                    )
                )
                print("  rate: {:.2%}".format(1.0 * corr / all_ans))

            if is_splice:
                print("\nocr token contain answer")
                corr = 0
                all_ans = 0
                for context in _set["data"]:
                    answers = [self.norm_answer(ans) for ans in context["answers"]]
                    ocr_tokens = [
                        self.format_text(token) for token in context["ocr_tokens"]
                    ]
                    if is_spell_check:
                        ocr_tokens = [
                            self.spell_check_single_word(w) for w in ocr_tokens
                        ]
                    all_ans += 1
                    for ans in answers:
                        for word in ocr_tokens:
                            if ans in word:
                                corr += 1
                                break
                        else:
                            continue
                        break

                print(
                    "# all answers: {}\t# ocr tokens contains answer: {}".format(
                        all_ans, corr
                    )
                )
                print("  rate: {:.2%}".format(1.0 * corr / all_ans))

            if is_over_one:
                print("\nmore than one ocrs token contain answer")
                corr = 0
                all_ans = 0
                for context in _set["data"]:
                    answers = [self.norm_answer(ans) for ans in context["answers"]]
                    ocr_tokens = [
                        self.format_text(token) for token in context["ocr_tokens"]
                    ]
                    if is_spell_check:
                        ocr_tokens = [
                            self.spell_check_single_word(w) for w in ocr_tokens
                        ]
                    all_ans += 1
                    for ans in answers:
                        temp = 0
                        for word in ocr_tokens:
                            if ans in word:
                                temp += 1
                                if temp == 2:
                                    corr += 1
                                    break
                        else:
                            continue
                        break

                print(
                    "# all answers: {}\t# ocr tokens contains answer: {}".format(
                        all_ans, corr
                    )
                )
                print("  rate: {:.2%}".format(1.0 * corr / all_ans))


if __name__ == "__main__":
    TVQA = Stat()
    TVQA.stat_zero_ocr_token()
    TVQA.stat_corr_answer()

