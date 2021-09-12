from celery import Celery

app = Celery('server', backend='redis://127.0.0.1:6379/0', broker='redis://127.0.0.1:6379/0')
app.conf.broker_url = 'redis://127.0.0.1:6379/0'
app.conf.result_backend = 'redis://127.0.0.1:6379/0'


@app.task
def add(x, y):
    return x + y
