import torch
import sqlite3
import hashlib
from celery import Celery
import pathlib

from celery.result import AsyncResult
from flask import Flask, send_file, redirect
from flask import request
from torchvision.utils import save_image

from models.my_simple_vae import VAE
from utils.is_in_docker import is_in_docker

print('Inside docker:', is_in_docker())
print('Script running at', pathlib.Path().resolve())
app = Flask(__name__, static_url_path='')  # Main object of the Flask application

MODEL_PATH = "./my_trained_models/10epochs_my_vae.pth"
device = torch.device("cpu")
model = VAE(device=device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

db = sqlite3.connect('./db/cache.db', check_same_thread=False)

img_to_task_id = {}
celery_app = Celery('server', backend='redis://redis', broker='redis://redis')
# celery_app = Celery('server', backend='redis://127.0.0.1:6379/0', broker='redis://127.0.0.1:6379/0')
# celery_app.conf.broker_url = 'redis://127.0.0.1:6379/0'
# celery_app.conf.result_backend = 'redis://127.0.0.1:6379/0'


@app.route('/')  # Function handler for /
def hello():
    return send_file('./templates/index.html')  # "Hello, from Flask"  # Return the string as a response


@app.route('/redirect_to_image_generation', methods=['POST'])  # Function handler for /redirect_to_image_generation
def image_gen_entry_point():
    name_of_image = request.values['image_name']
    if len(name_of_image) > 0:
        print('redirecting...')
        return redirect('/image/' + name_of_image)
    return redirect('/')


@app.route("/image/<image_id>")
def frequency_check_handler(image_id):
    cursor = db.execute('SELECT filename from CACHE WHERE id = ?', [image_id])
    rows = cursor.fetchall()
    print('rows', rows)
    cursor.close()
    if len(rows) == 1:
        filename = rows[0][0]
        filepath = pathlib.PurePath('./temp', filename)
        return send_file(filepath)  # 'Picture in database'
    elif image_id in img_to_task_id:
        task_id = img_to_task_id[image_id]
        task = AsyncResult(task_id, app=celery_app)
        print('task state', task)
        if task.ready():
            print('task is ready')
            filename = task.result
            db.execute('REPLACE INTO CACHE VALUES (?, ?)', [image_id, filename])
            return 'Thank you for using my "Blindr 2.0", please refresh the page to see Neural Network generated ' \
                   'picture! '
        cursor = db.execute('SELECT filename from CACHE')
        all_rows = cursor.fetchall()
        cursor.close()
        return 'running ' + task_id + ', please refresh page!     DEBUG info, rows = ' + str(rows) + ' allrows = ' + str(all_rows)
    else:
        task = generate_image.delay(image_id)
        # generate_image(image_id)
        img_to_task_id[image_id] = task.id
        return f'Picture "{image_id}" is not in our collection, we generating it with task number ' + img_to_task_id[image_id]


@celery_app.task
def generate_image(image_id):
    print('Hello from generate_image')
    try:
        print('sampling started')
        images = model.sample(1).cpu()
        print('sampling finished')
        img1 = images[0]
    except Exception as ex:
        print('exception happened')
        print(ex)
    #db.execute('REPLACE INTO CACHE VALUES (?, ?)', [1, 1])

    new_name = hashlib.sha256(image_id.encode()).hexdigest() + '.png'
    save_image(img1, './temp/' + new_name)
    print('image saved')

    #db.execute('REPLACE INTO CACHE VALUES (?, ?)', [image_id, new_name])
    return new_name


if __name__ == '__main__':
    app.run("0.0.0.0", 8000)  # Run the server on port 8000
