import logging, argparse, json, os
from flask import Flask, request, jsonify
from classificator_taxtags import search, init

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def rules_apply():
    quest_json = request.get_json()
    # print(quest_json)
    data_rout = r"./data/tax_dems_jsons"
    with open(os.path.join(data_rout, str(quest_json['docid'])+".json"), 'w', encoding='utf8') as f:
        json.dump(quest_json, f)
    res = search(quest_json['docid'], quest_json['text'])
    return jsonify(res)


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    init()
    # global data_rout
    # data_rout = r"./data/tax_dems_jsons"

    args = argparse.ArgumentParser()
    args.add_argument('--host', dest='host', default='0.0.0.0')
    args.add_argument('--port', dest='port', default='4888')
    args = args.parse_args()
    app.run(host=args.host, port=args.port)
