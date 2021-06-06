import stanza

nlp = stanza.Pipeline(lang='uk', processors='tokenize,mwt,pos,lemma', tokenize_no_ssplit=True)

punctuation = [".", ",", "(", ")", "!", "?", ":", ";", "-"]

with open("anglicisms", 'r') as f:
    en_dict = [x.lower() for x in f.read().split()]

with open("dictionary_sum_clean.txt", 'r') as f:
    uk_dict = f.read().split()


def get_lemmas(text):
    list_of_lemmas = []
    for sent in nlp(text).sentences:
        for word in sent.words:
            lemma = word.lemma
            if lemma not in list_of_lemmas and lemma not in punctuation and ("'" in lemma or lemma.isalpha()):
                list_of_lemmas.append(lemma.lower())
    return list_of_lemmas


def detect_neologisms(words):
    anglicisms = []
    new_words = []
    for word in words:
        if word in en_dict:
            anglicisms.append(word)
        if word not in uk_dict and word not in en_dict:
            new_words.append(word)
    return anglicisms, new_words


print(detect_neologisms(get_lemmas("""Сьогодні відбулися конкурсні випробування «Сам собі іміджмейкер» (захист постеру)
та «Публічний виступ у стилі TED». Чекаємо на презентації інших учасників і результати""")))
