# здесь будут функции, которые нужны для других объектов
import os, pickle, difflib, logging, re
from abc import ABC, abstractmethod
from itertools import groupby


# данный класс (или функция) должен знать все форматы моделей и уметь их загружать
def model_load(incoming_model):
    names = [nm for nm in incoming_model]
    for nm in names:
        assert nm in ["model_name", "model_type",
                      "etalons", "texts", "coeff", "tags", "lingvo", "classificator_algorithms",
                      "texts_algorithms", "tokenizer", "simple_tokenizer"], \
            "наименования входящей модели не соответствуют ожиданию класса Loader"

    assert incoming_model["model_type"] in ["siamese_lstm_d2v", "simple_rules", "lsi",
                                            "intersec_share", "simple_tokenizer"], \
        "тип модели не соответствует ожиданию класса Loader"

    etalons_dict_names = [nm for nm in incoming_model["etalons"]]
    # возвращает эталоны, к которым применяется правило (проверяет их на соответствие соглашению)
    for nm in etalons_dict_names:
        assert nm in ["rules", "texts", "coeff", "tags"], (
            "имена словаря etalons не соответствуют ожиданиям класса Loader")

    return incoming_model


class Loader():
    def __init__(self, incoming_model):
        self.in_model = model_load(incoming_model)
        # возвращает ключ модели (имя модели) по ключу остальные объекты "понимают" что за функции им нужно запускать
        self.model_type = self.in_model["model_type"]
        # возвращает модели для правил
        self.classificator_algorithms = self.in_model["classificator_algorithms"]
        # возвращает модели для обработки текстов (например, Word2Vec - модели векторизации)
        self.texts_algorithms = self.in_model["texts_algorithms"]
        # возвращает словари для обработки текста
        self.dictionaries = self.in_model["lingvo"]
        # возвращает тип токенезации входящего текста
        self.tokenizer_type = self.in_model["tokenizer"]
        # возвращает правила
        self.application_field = self.in_model["etalons"]

    # загрузка модели с проверкой имен:


class AbstractRules(ABC):
    def __init__(self):
        # перменная, описывающая, какие модели входят в класс
        self.model_types = []

    # должен возвращать структуру типа: [(num, [(tag, True), ...]), ...]
    # num - номер текста
    # tag - номер текста
    # True / False - результат для данного тега и данного текста
    @abstractmethod
    def rules_apply(self, text: []):
        pass


# Основное время тратится на загрузку лемматизатора и на лемматизацию эталонов
# classes - классы, которые используются в цепочке
# функция, применяющая набор моделей (цепочку моделей) к входящему тексту
# допущение - модели должны содержать одинаковые эталоны с одинаковыми тегами
# models :[] -> [loader_obj, ...]
# true_tags = classes_models[0][1].application_field["tags"]
# приоритет моделей соответствует последовательности загрузки моделей в класс
class ModelsChain(AbstractRules):
    def __init__(self, models_classes=[]):
        self.models_classes = models_classes
        self.classes_models = self.classes_modles_fill()

    def classes_modles_fill(self):
        classes_with_model = []
        for Cls, model in self.models_classes:
            try:
                classes_with_model.append(Cls(Loader(model)))
            except Exception as exx:
                logging.error('unable to link classes to models: "{}" to "{}"'.format(Cls, model.model_type))
                logging.error(exx)
        return classes_with_model

    def rules_apply(self, texts):  # выбор классов для полученных моделей:
        results = []
        for Class_with_model in self.classes_models:
            cls_results = Class_with_model.rules_apply(texts)
            for tx_result in cls_results:
                results.append(tx_result)
        # grouping results with the same texts
        results_grouped = [(x, [z[1] for z in y]) for x, y in
                           groupby(sorted(results, key=lambda x: x[0]), key=lambda x: x[0])]

        tags_result = []
        if len(self.models_classes) == 1:
            for tx_num, txt_tags_group in results_grouped:
                if txt_tags_group != [[]]:
                    tags_result.append(tuple((tx_num, txt_tags_group[0])))
                else:
                    tags_result.append(tuple((tx_num, None)))
        else:
            for tx_num, txt_tags_group in results_grouped:
                tags_result.append(tuple((tx_num, decision_choice(txt_tags_group))))
        return tags_result


# совсем утилитарная функция для класса ModelsChain
# выдает True только для тех правил, для которых все алгоритмы в цепочке сработали True
def decision_choice(ls: []):
    for i in ls[0]:
        decision = False
        for l in ls[1:]:
            if i in l:
                decision = True
            else:
                decision = False
        if decision:
            return i
    return None


""" Утилитарные функции над массивами """
""" Оставляет только базовые элементы сложной (иерархической) структуры """


def flatten_list(iterable):
    for elem in iterable:
        if not isinstance(elem, list):
            yield elem
        else:
            for x in flatten_list(elem):
                yield x


def flatten_tuple(iterable):
    for elem in iterable:
        if not isinstance(elem, tuple):
            yield elem
        else:
            for x in flatten_tuple(elem):
                yield x


"""Нарезает массив окном размера len c шагом stride"""


def sliceArray(src: [], length: int = 1, stride: int = 1):
    return [src[i:i + length] for i in range(0, len(src), stride) if len(src[i:i + length]) == length]


'''Нарезает массив окном размера len с шагом в размер окна'''


def splitArray(src: [], length: int):
    return sliceArray(src, length, length)


# Преобразует массив токенов в мешок (каждый токен представлен кортежем
# -- (токен, сколько раз встречается в массиве))
def arr2bag(src: []):
    return [(x, src.count(x)) for x in set(src)]


"""Возвращает массив токенов src за исключением rem"""


def removeTokens(src: [], rem: []):
    return [t for t in src if t not in rem]


"""Заменяет в массиве src множество токенов аскриптора asc дексрипторами токена (синонимия)"""


def replaceAscriptor(src: [], asc: [], desc: []):
    src_repl = []
    length = len(asc)
    src_ = [src[i:i + length] for i in range(0, len(src), 1)]
    i = 0
    while i < len(src_):
        if src_[i] == asc:
            src_repl = src_repl + desc
            i += length
        else:
            src_repl.append(src_[i][0])
            i += 1
    return src_repl


""" Прочие функции """
""" функция оценивающая похожесть строк (возвращает оценку похожести) """


def strings_similarities(str1: str, str2: str):
    return difflib.SequenceMatcher(None, str1, str2).ratio()


# сравнение двух списков (возвращает токены, принадлежащие обоим спискам)
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


# функция поиска между заданными паттернами
def search_between_patterns(pattern1, pattern2, lem_tx):
    return re.findall(pattern1 + '.*?' + pattern2, lem_tx)


# df["text"]
# лемматизация текстов датафрейма
# columns_for_change_names - список столбцов (имен столбцов) датафрейма, которые должны быть обработаны
# changed_columns_names - список столбцов (имен столбцов) датафрейма после обработки

"""
def df_lemmatize(df, columns_for_change_names = ["text"], changed_columns_names = ["changed_text"]):
    assert(len(columns_for_change_names) == len(changed_columns_names))
    for column_for_ch, ch_column in zip(columns_for_change_names, changed_columns_names):
        df[ch_column] = df[column_for_ch].apply(lambda tx: texts_lemmatize([tx]))
    return df


# замена словарей текстов датафрейма
# columns_for_change_names : [[]]
#asc_dsc_tuples_list : []
def df_processing(df, asc_dsc_tuples_list, columns_for_change_names = ["text"], changed_columns_names = ["changed_text"]):
    assert(len(columns_for_change_names) == len(changed_columns_names))
    for column_for_ch, ch_column in zip(columns_for_change_names, changed_columns_names):
        df[ch_column] = df[column_for_ch]
        for asc_dsc_tuple in asc_dsc_tuples_list:
            df[ch_column] = df[ch_column].apply(lambda tx: texts_asc_dsc_change(asc_dsc_tuple, tx))
    return df
"""

if __name__ == "__main__":
    import time

    data_rout = r'./data'
    models_rout = r'./models'

    from classificators import SimpleRules

    with open(os.path.join(models_rout, "tax_tags", "simple_rules_model.pickle"), "br") as f:
        model = pickle.load(f)

    smplm = ModelsChain([(SimpleRules, model)])

    txs = ["в ходе проведения камеральной налоговой проверки на основе Налоговая декларация по налогу на прибыль "
           "организаций доходы от реализации по данным деклараций снижение уменьшение по сравнению ранее "]

    print(smplm.rules_apply(txs))
