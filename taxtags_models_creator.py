import os, pickle
import pandas as pd

data_rout = r'./data'
models_rout = r'./models'

lingv_rules_df = pd.read_csv(os.path.join(data_rout, 'tax_dem_data', 'lingv_rules2.csv'))
ngrams_df = pd.read_csv(os.path.join(data_rout, 'tax_dem_data', 'ngrams.csv'))

# подготовка списка синонимов:
sinonims_files = ['sinonims.csv']
sinonims = []
for file_name in sinonims_files:
    sin_df = pd.read_csv(os.path.join(data_rout, 'tax_dem_data', file_name))
    sinonims.append(list(zip(sin_df["words"], sin_df["initial_forms"])))

# ========== simple rules model ======================
model = {"model_name": "kosgu_include_and_model",
         "model_type": "simple_rules",
         "etalons": {
             "rules": list(lingv_rules_df["rules"]),
             "texts": list(lingv_rules_df["words"]),
             "tags": list(lingv_rules_df["tag"]),
             "coeff": list(lingv_rules_df["coeff"])},
         "lingvo": [{"synonyms": sinonims, "tokenize": True},
                    {"ngrams": [[]], "tokenize": False},
                    {"stopwords": [[]], "tokenize": True},
                    {"workwords": [[]], "tokenize": True}],
         "classificator_algorithms": {},
         "texts_algorithms": {},
         "tokenizer": "SimpleTokenizer"}

with open(os.path.join(models_rout, 'tax_tags', "simple_rules_model.pickle"), "bw") as f:
    pickle.dump(model, f)