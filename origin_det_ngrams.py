import nltk
from nltk.util import bigrams, trigrams, ngrams
import sqlite3
from getting_ngrams import test_set_en, test_set_uk

db = sqlite3.connect("ngramDB.db")
cur = db.cursor()

# test_set_en.remove("фокстер'єр")


def detect_origin(word):
    ngram_list = []
    for i in range(3, 7):
        ngram = list(ngrams(word, i))
        ngram_list.extend(ngram)
    score_en = 0
    score_uk = 0
    for ngram in ngram_list:
        points_en = (cur.execute("""SELECT internal_fr FROM ngram_2 WHERE ngram = \"""" + str(ngram) +
                                 """\" AND origin = "en";""")).fetchall()
        if len(points_en) != 0:
            score_en += points_en[0][0]
        points_uk = (cur.execute("""SELECT internal_fr FROM ngram_2 WHERE ngram = \"""" + str(ngram) +
                                 """\" AND origin = "uk";""")).fetchall()
        if len(points_uk) != 0:
            score_uk += points_uk[0][0]
    print(score_uk, score_en)
    if score_en > score_uk:
        origin = "english"
    else:
        origin = "ukrainian"
    return origin


def get_results(test_set):
    with open("ngram_test_2", 'a') as f:
        for word in test_set:
            origin = detect_origin(word)
            f.write("{0} : {1}\n".format(word, origin))


# get_results(test_set_en)
# get_results(test_set_uk)
print(detect_origin("результат"))
