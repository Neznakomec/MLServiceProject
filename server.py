import torch
import sqlite3
import hashlib
from celery import Celery
import pathlib
from flask import Flask, send_file
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


@app.route('/')  # Function handler for /
def hello():
    # generate image
    # images = model.sample(1).cpu()
    # img1 = images[0]
    # save_image(img1, 'img1.png')

    # send image to user
    # filename = 'img1.png'
    # return send_file(filename, mimetype='image/png')

    return send_file('./templates/index.html')  # "Hello, from Flask"  # Return the string as a response


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
        cursor = db.execute('SELECT filename from CACHE')
        all_rows = cursor.fetchall()
        cursor.close()
        return 'running ' + task_id + ' rows = ' + str(rows) + ' allrows = ' + str(all_rows)
    else:
        # task = AsyncResult(task_id, app=celery_app)
        task = generate_image.delay(image_id)
        # generate_image(image_id)
        img_to_task_id[image_id] = task.id
        return 'Picture unknown, created task ' + img_to_task_id[image_id]


@celery_app.task
def generate_image(image_id):
    db.execute('REPLACE INTO CACHE VALUES (?, ?)', [1, 1])
    images = model.sample(1).cpu()
    img1 = images[0]

    new_name = hashlib.sha256(image_id.encode()).hexdigest() + '.png'
    save_image(img1, './temp/' + new_name)

    db.execute('REPLACE INTO CACHE VALUES (?, ?)', [image_id, new_name])
    return 0


if __name__ == '__main__':
    app.run("0.0.0.0", 8000)  # Run the server on port 8000
