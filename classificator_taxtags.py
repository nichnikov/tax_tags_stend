import os, pickle
from texts_processors import TokenizerApply
from classificators import SimpleRules
from utility import search_between_patterns, Loader, ModelsChain


# m = None
# answers_tags = None
# answers_df = None


def init():
    # тут поднять всякие модели и пр.
    models_rout = os.path.dirname(__file__)

    # загрузка моделей для каждого pubid и формирование словаря, в котором pubid являются ключами, 
    # значениями классифицирующие модели и связки тегов с айди ответов (и модулей)
    models_dict = {"simple_rules_model": None}

    # загрузка моделей и превращение их сразу в loader_obj
    for model_name in models_dict:
        with open(os.path.join(models_rout, 'models/tax_tags', str(model_name) + ".pickle"), "br") as f:
            model = pickle.load(f)
            models_dict[model_name] = model

    # загрузим лемматизатор для паттернов:
    tknz = TokenizerApply(Loader(models_dict["simple_rules_model"]))

    global pattern1, pattern2
    # лемматизируем паттерны для обора фрагментов
    pattern1 = tknz.texts_processing(["в ходе проведения"])[0]
    pattern2 = tknz.texts_processing(["В течение 5 <4> рабочих дней"])[0]

    """определение моделей, которые потом используются для разных pubid"""
    model_1 = ModelsChain([(SimpleRules, models_dict["simple_rules_model"])])

    global pub_models
    pub_models = {1: {"model": model_1, "tag_answ_link": None, "tokenizer": tknz}}
    # return None


# функция для одного текста (модели могут отрабатывать на пакете (списке) текстов)
def search(doc_id, tx: str, pbid=1):
    lem_tx_list = pub_models[pbid]["tokenizer"].texts_processing([tx])[0]
    txt = search_between_patterns(" ".join(pattern1), " ".join(pattern2), " ".join(lem_tx_list))
    try:
        true_results = pub_models[pbid]["model"].rules_apply(txt)
        return {"id": doc_id, "tag": true_results[0][1]}
    except:
        return None


if __name__ == '__main__':
    init()
    models_rout = os.path.dirname(__file__)
    with open(os.path.join(models_rout, 'models/tax_tags', "simple_rules_model" + ".pickle"), "br") as f:
        model_tax = pickle.load(f)

    for i in model_tax:
        print(i)

    print("model_type:", model_tax["model_type"])

    print("model_type:", Loader(model_tax).model_type)

    clss = SimpleRules(Loader(model_tax))
    # print(clss.tknz_model.application_field)

    print(clss.tknz_model.application_field)

    print("pattern1:", pattern1, "pattern2:", pattern2)

    data_rout = r"./tax_demands"

    with open("example.txt", "r") as f:
        tx = f.read()

    # print(tx)

    lem_tx_list = pub_models[1]["tokenizer"].texts_processing([tx])[0]
    txt = search_between_patterns(" ".join(pattern1), " ".join(pattern2), " ".join(lem_tx_list))
    print(txt)

    clss = ModelsChain([(SimpleRules, model_tax)])
    ldobj = Loader(model_tax)
    print(ldobj.application_field)
    print(clss.classes_models)

    print("txt:", txt)
    print("clss.rules_apply:", clss.rules_apply([" ".join(txt)]))

    res = search("111", " ".join(txt), pbid=1)
    print("search:", res)