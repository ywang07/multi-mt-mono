'''
Usage:
"python preprocess_mono.py monolingual_file"
Monolingual data filtering hard rules, including:
1. duplicated sentences remove
2. sentences with #words < 3 or > 500
3. sentences with '/', '|', '-' > 10, #punc
4. sentences with punctuations/characters > 0.5
# 5. sentences contains word composed by more than 50 charcters
# 6. sentences with average characs for word > 50 or < 3
7. sentences with punctuations > 30
8. sentences with html address and html tags
# 9. language id
# 9. [optional]: non english characters > 0.25
'''
import sys
import re
import argparse
import langid
from string import punctuation

parser = argparse.ArgumentParser()
parser.add_argument('src', help='source file')
parser.add_argument('--lid', default=None)
parser.add_argument('--dedup', action='store_true', default=False, help="do dedup only")
parser.add_argument('--soft_html', action='store_true', default=False,
                    help='whether to use soft version only to remove html tag, not the sentence')
args = parser.parse_args()
f1 = args.src

min_tok = 3
max_top = 500
max_special_punc = 10
ratio_special_punc = 0.5
avg_word_len_lb = 3
avg_word_len_ub = 50
max_char_per_word = 50
punc_max_num = 30
lattin_ratio = 0.3


# Duplicated sentences remove
def dup_remove(x_in):
    all_lines = [x.strip() for x in x_in]
    x_out = set(all_lines)

    print('After removing duplicated sentences, remain %i sentences' % len(x_out))
    return x_out


def sentence_word_num_remove(x_in):
    def check_word_num(sent):
        segs = sent.strip().split()
        if len(segs) < min_tok or len(segs) > max_top:
            return False
        return True

    x_out = []

    for x in x_in:
        if check_word_num(x):
            x_out.append(x.strip())

    print('After removing sentences with too less or too many words, remain %i sentences' % len(x_out))
    return x_out


# Specific punctuation number exceeded sentence remove
def specfic_punc_remove(x_in):
    def hot_fix_filter(sent):
        sent = sent.strip()
        if sent.count("/") > max_special_punc:
            return False
        if sent.count("|") > max_special_punc:
            return False
        if sent.count("-") > max_special_punc:
            return False
        if len(re.findall("[\d\-\|/]", sent)) / len(sent) > ratio_special_punc:
            return False
        return True

    x_out = []

    for x in x_in:
        if hot_fix_filter(x):
            x_out.append(x.strip())

    print('After removing sentences with too many specific punctuations, remain %i sentences' % len(x_out))
    return x_out


# Characters condition remove
def characs_remove(x_in):
    def filter_by_len(sent):
        segs = sent.strip().split()
        for x in segs:
            if len(x) > max_char_per_word:
                return False
        m_char = sum([len(x) for x in segs])
        m_word = len(segs)
        ratio = m_char * 1. / (m_word + 1e-9)
        if ratio > avg_word_len_ub or ratio < avg_word_len_lb:
            return False
        return True

    x_out = []

    for x in x_in:
        if filter_by_len(x):
            x_out.append(x.strip())

    print('After removing sentence with characters condition, remain %i sentences' % len(x_out))
    return x_out


# Punctuation condition remove
def punctuation_remove(x_in):
    x_out = []

    count_func = lambda l1, l2: sum([1 for x in l1 if x in l2])

    punctuation_set = set(punctuation)
    for x in x_in:
        m_punc_x = count_func(x.strip(), set(punctuation_set))
        if m_punc_x / (len(x.strip()) + 1e-9) > ratio_special_punc or m_punc_x > punc_max_num:
            continue
        x_out.append(x.strip())

    print('After removing sentences with too much punctuations, remain %i sentences' % len(x_out))
    return x_out


# Html address or tags contained sentence remove
def html_remove(x_in):
    x_out = []

    def filter_by_html(sentence):
        sen = sentence.strip()
        detector = re.compile('<.*?>')
        html_tag = re.findall(detector, sen)
        if html_tag or 'https://' in sen or 'http://' in sen:
            return False
        return True

    def soft_filter_by_html(sent):
        sent = sent.strip()
        detector = re.compile('<.*?>')
        sent = re.sub(detector, '', sent)
        sent = re.sub('https?:\/\/.*[ \r\n]', '', x, flags=re.MULTILINE)
        return sent

    for x in x_in:
        if args.soft_html:
            x_out.append(soft_filter_by_html(x))
        else:
            if filter_by_html(x):
                x_out.append(x.strip())

    print('After removing sentences with html address or tags, remain %i sentences' % len(x_out))
    return x_out


# Optional: Lattin letter contained sentence remove
def lattin_remove(x_in):
    def count_lattin(sent):
        if len(re.findall("[^a-zA-Z]", sent)) / len(sent) > lattin_ratio:
            return False
        return True

    x_out = []
    for x in x_in:
        if count_lattin(x.strip()):
            x_out.append(x.strip())

    print('After removing sentences with too much lattin characs, remain %i sentences' % len(x_out))
    return x_out


def language_id(x_in, lid):
    x_out = []
    for x in x_in:
        if langid.classify(x)[0] == lid:
            x_out.append(x)

    print("After running langugage detector, remain %i sentences" % len(x_out))
    return x_out


filter_1 = []

fr_1 = open(f1, "r", encoding="utf8")

f1_all_lines = fr_1.readlines()
print("total {} lines loaded".format(len(f1_all_lines)))

filter_1 = dup_remove(f1_all_lines)
if not args.dedup:
    filter_1 = sentence_word_num_remove(filter_1)
    filter_1 = specfic_punc_remove(filter_1)
    # filter_1  = characs_remove(filter_1)
    filter_1 = punctuation_remove(filter_1)
    filter_1 = html_remove(filter_1)
    # filter_1  = lattin_remove(filter_1)

if args.lid is not None:
    filter_1 = language_id(filter_1, lid=args.lid)

fr_1.close()

fw_1 = open(f1 + ".clean", "w", encoding="utf8")

print('After all filtering rules, remain %i sentences' % len(filter_1))

for x in filter_1:
    print(x, file=fw_1)

fw_1.close()