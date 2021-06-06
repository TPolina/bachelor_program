import nltk
from nltk.util import bigrams, trigrams, ngrams
from collections import Counter
import sqlite3

db = sqlite3.connect("ngramDB.db")
cur = db.cursor()

with open("anglicisms", 'r') as f:
    en_words = [x.lower() for x in f.read().split()]
with open("ukr_words", 'r') as f:
    uk_words = [x.lower() for x in f.read().split()]

test_set_en = en_words[::10]
for word in en_words:
    if word in test_set_en:
        en_words.remove(word)
test_set_uk = uk_words[::10]
for word in uk_words:
    if word in test_set_uk:
        uk_words.remove(word)


def get_ngram_dict(words_list):
    ngram_list = []
    for word in words_list:
        for i in range(3, 7):
            ngram = list(ngrams(word, i))
            ngram_list.extend(ngram)
    ngram_dict = nltk.FreqDist(ngram_list)
    return ngram_dict


def remove_single_ngram(freq_dict):
    new_dict = {}
    for k, v in freq_dict.items():
        if v > 1:
            new_dict[k] = v
    return new_dict


en_dict = remove_single_ngram(dict(get_ngram_dict(en_words)))
uk_dict = remove_single_ngram(dict(get_ngram_dict(uk_words)))

total = {k: en_dict.get(k, 0) + uk_dict.get(k, 0) for k in set(en_dict) | set(uk_dict)}


def fill_db(freq_dict, lang):
    for key, value in freq_dict.items():
        cur.execute("""INSERT INTO ngram_2 (origin, ngram, total_in_dict, total, internal_fr, overall_fr)
                    VALUES(?, ?, ?, ?, ?, ?);""", (lang, str(key), value, total[key],
                                                   value / len(freq_dict), value / len(total),))


fill_db(en_dict, "en")
fill_db(uk_dict, "uk")

db.commit()
db.close()
