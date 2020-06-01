# здесь будут объекты для создания правил
# https://stackoverflow.com/questions/47115946/tensor-is-not-an-element-of-this-graph
# https://kobkrit.com/tensor-something-is-not-an-element-of-this-graph-error-in-keras-on-flask-web-server-4173a8fe15e1

import os, pickle, logging
from utility import Loader, AbstractRules, ModelsChain, intersection, strings_similarities  # contrastive_loss
from texts_processors import TokenizerApply
from itertools import groupby
# from gensim.similarities import Similarity
from gensim.similarities import MatrixSimilarity


def include_and(tokens_list, text_list, coeff=0.0):
    for token in tokens_list:
        if token not in text_list:
            return False
    return True


def include_or(tokens_list, text_list, coeff=0.0):
    for token in tokens_list:
        if token in text_list:
            return True
    return False


def exclude_and(tokens_list, text_list, coeff=0.0):
    for token in tokens_list:
        if token in text_list:
            return False
    return True


def exclude_or(tokens_list, text_list, coeff=0.0):
    for token in tokens_list:
        if token not in text_list:
            return True
    return False


def intersec_share(tokens_list, text_list, intersec_coeff=0.7):
    intersec_tks = intersection(tokens_list, text_list)
    if len(intersec_tks) / len(tokens_list) > intersec_coeff:
        return True
    else:
        return False


def include_str(tokens_str, text_str, coeff=0.0):
    if tokens_str in text_str:
        return True
    else:
        return False


def exclude_str(tokens_str, text_str, coeff=0.0):
    if tokens_str not in text_str:
        return True
    else:
        return False


def include_str_p(tokens_list: list, txt_list: list, coeff):
    length = len(tokens_list)
    txts_split = [txt_list[i:i + length] for i in range(0, len(txt_list), 1) if
                  len(txt_list[i:i + length]) == length]
    for tx_l in txts_split:
        if strings_similarities(' '.join(tokens_list), ' '.join(tx_l)) >= coeff:  # self.sims_score:
            return True
    return False


def exclude_str_p(tokens_list: list, txt_list: list, coeff):
    length = len(tokens_list)
    txts_split = [txt_list[i:i + length] for i in range(0, len(txt_list), 1) if
                  len(txt_list[i:i + length]) == length]
    for tx_l in txts_split:
        if strings_similarities(' '.join(tokens_list), ' '.join(tx_l)) >= coeff:  # self.sims_score:
            return False
    return True


class SimpleRules(AbstractRules):
    def __init__(self, loader_obj):
        self.model_types = [("simple_rules", None)]
        self.functions_dict = {"include_and": include_and, "include_or": include_or,
                               "exclude_and": exclude_and, "exclude_or": exclude_or,
                               "include_str": include_str, "include_str_p": include_str_p,
                               "exclude_str_p": exclude_str_p, "intersec_share": intersec_share}
        self.model = loader_obj
        self.tokenizer = TokenizerApply(self.model)
        self.tknz_model = self.tokenizer.model_tokenize()

    def rules_apply(self, texts):
        decisions = []
        # применим правило к токенизированным текстам:
        for num, tknz_tx in enumerate(self.tokenizer.texts_processing(texts)):
            decisions_temp = []
            model_params = list(zip(self.tknz_model.application_field["tags"],
                                    self.tknz_model.application_field["rules"],
                                    self.tknz_model.application_field["texts"],
                                    self.tknz_model.application_field["coeff"]))
            # grouping rules with the same tag
            model_params_grouped = [(x, list(y)) for x, y in
                                    groupby(sorted(model_params, key=lambda x: x[0]), key=lambda x: x[0])]
            # оценка результатов применения правил для каждого тега (в каждой группе):
            for group, rules_list in model_params_grouped:
                decision = True
                for tg, rule, tknz_etalon, coeff in rules_list:
                    decision = decision and self.functions_dict[rule](tknz_etalon, tknz_tx, coeff)
                    # print(tg, rule, decision)
                # будем возвращать только сработавшие правила (True)
                if decision:
                    # decisions_temp.append((group, decision))
                    decisions_temp.append(group)
            decisions.append((num, decisions_temp))
        return decisions

    def rules_apply_debugging(self, texts):
        decisions = []
        # применим правило к токенизированным текстам:
        for num, tknz_tx in enumerate(self.tokenizer.texts_processing(texts)):
            decisions_temp = []
            model_params = list(zip(self.tknz_model.application_field["tags"],
                                    self.tknz_model.application_field["rules"],
                                    self.tknz_model.application_field["texts"],
                                    self.tknz_model.application_field["coeff"]))
            # grouping rules with the same tag
            model_params_grouped = [(x, list(y)) for x, y in
                                    groupby(sorted(model_params, key=lambda x: x[0]), key=lambda x: x[0])]
            # оценка результатов применения правил для каждого тега (в каждой группе):
            for group, rules_list in model_params_grouped:
                decision = True
                for tg, rule, tknz_etalon, coeff in rules_list:
                    decision = decision and self.functions_dict[rule](tknz_etalon, tknz_tx, coeff)
                    # print(tg, rule, decision)
                # будем возвращать только сработавшие правила (True)
                # if decision:
                    # decisions_temp.append((group, decision))
                    decisions_temp.append((group, decision))
            decisions.append((num, decisions_temp))
        return decisions


class LsiClassifier(AbstractRules):
    def __init__(self, loader_obj):
        self.model_types = [("lsi", None)]
        self.model = loader_obj
        self.tknz = TokenizerApply(self.model)
        self.tkz_model = self.tknz.model_tokenize()
        self.et_vectors = self.tkz_model.application_field["texts"]
        self.coeffs = self.tkz_model.application_field["coeff"]
        self.tags = self.tkz_model.application_field["tags"]
        # self.index = Similarity(None, self.et_vectors, num_features=self.model.texts_algorithms["num_topics"])
        self.index = MatrixSimilarity(self.et_vectors, num_features=self.model.texts_algorithms["num_topics"])

    def rules_apply(self, texts):
        text_vectors = self.tknz.texts_processing(texts)
        texts_tags_similarity = []
        # true_scores_results = []
        for num, text_vector in enumerate(text_vectors):
            # true_scores = [(tg, scr, True) for tg, scr, cf in list(zip(self.tags, self.index[text_vector],
            #                                                           self.coeffs)) if scr > cf]
            trues_list_scores = [(tg, scr, cf) for tg, scr, cf in list(zip(self.tags, self.index[text_vector],
                                                                           self.coeffs)) if scr > cf]
            # отсортируем, чтобы выводить наиболее подходящие результаты (с наибольшим скором)
            trues = [tg for tg, scr, cf in sorted(trues_list_scores, key=lambda x: x[1], reverse=True)]

            # falses = [(tg, False) for tg, scr, cf in list(zip(self.tags, self.index[text_vector], self.coeffs))
            #         if scr < cf]
            # texts_tags_similarity.append((num, trues + falses))
            texts_tags_similarity.append((num, trues))
            # print(texts_tags_similarity)
            # true_scores_results.append(true_scores)
        #return texts_tags_similarity, true_scores_results
        return texts_tags_similarity



if __name__ == "__main__":
    import time

    data_rout = r'./data'
    models_rout = r'./models'

    with open(os.path.join(models_rout, "fast_answrs", "bss_lsi_model.pickle"), "br") as f:
        model = pickle.load(f)

    cl = LsiClassifier(Loader(model))
    tx = "коронавирус самоизоляция анапа"
    # tx = "спецпропуска Омская область"
    rls = cl.rules_apply([tx])
    print(rls)
    # print(scrs)

'''
    mc = ModelsChain([Loader(model)], classes=[SimpleRules, LsiClassifier])
    tx = "командировки статья косгу"
    t1 = time.time()
    print(mc.rules_apply([tx]), time.time()-t1)

    with open(os.path.join(models_rout, "fast_answrs", "kosgu_incl_and_test_model.pickle"), "br") as f:
        kosgu_incl_and = pickle.load(f)

    with open(os.path.join(models_rout, "fast_answrs", "bss_lsi_model.pickle"), "br") as f:
        bss_lsi = pickle.load(f)

    with open(os.path.join(models_rout, "fast_answrs", "bss_intersec_share_model.pickle"), "br") as f:
        bss_intersec = pickle.load(f)

    with open(os.path.join(models_rout, "fast_answrs", "bss_include_and_model.pickle"), "br") as f:
        bss_include_and = pickle.load(f)

    tx = ["шпаргалка, чтобы определить квр и косгу для командировочных расходов госучреждений"]
    mdschain = ModelsChain([Loader(kosgu_incl_and)], classes=[SimpleRules, LsiClassifier])
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print(tx[0], "kosgu_incl_and:", rt_t, time.time() - t1)

    # tx = ["кто может применять упрощенный баланс"]
    tx = ["упрощенная финансовая отчетность кто сдает"]
    mdschain = ModelsChain([Loader(bss_lsi)], classes=[SimpleRules, LsiClassifier])
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print(tx[0], "bss_lsi:", rt_t, time.time() - t1)

    tx = ["кто может применять упрощенный баланс"]
    mdschain = ModelsChain([Loader(bss_intersec)], classes=[SimpleRules, LsiClassifier])
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print(tx[0], "bss_intersec:", rt_t, time.time() - t1)

    tx = ["кто может не применять ккт"]
    mdschain = ModelsChain([Loader(bss_include_and)], classes=[SimpleRules, LsiClassifier])
    t1 = time.time()
    rt_t = mdschain.rules_apply(tx)
    print(tx[0], "bss_include_and:", rt_t, time.time() - t1)
'''
