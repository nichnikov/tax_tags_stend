import logging, argparse, json, os
from classificator_taxtags import search, init
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = './data/uploadfiles'
# UPLOAD_FOLDER = '/home/alexey/big/github/tax_tags_stend/tax_tags_stend/data/uploadfiles'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# как перезапускать flask:
# https://dev.to/lukeinthecloud/python-auto-reload-using-flask-ci6
# https://gist.github.com/sethbunke/e901232e838a53cd1fe0becbf852c0e6
# код для загрузки файлов взят тут: https://flask-russian-docs.readthedocs.io/ru/latest/patterns/fileuploads.html


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("Hello World!")
            return redirect(url_for('upload_file', filename=filename))
            # return "OK"

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Загрузчик требований в формате pdf</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    # init()
    # global data_rout
    # data_rout = r"./data/tax_dems_jsons"
    app.run(host='0.0.0.0', port=5000)
