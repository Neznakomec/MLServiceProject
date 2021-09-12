FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

WORKDIR /app
EXPOSE 8000

RUN pip install Flask
RUN pip install celery[redis]
RUN pip install redis

COPY server.py /app/server.py

ADD db $HOME/app/db
ADD utils $HOME/app/utils
ADD models $HOME/app/models
ADD static $HOME/app/static
ADD templates $HOME/app/templates
ADD temp $HOME/app/temp

COPY ./my_trained_models/10epochs_my_vae.pth $HOME/app/my_trained_models/10epochs_my_vae.pth

#ENTRYPOINT ["bash"]
CMD ['celery', '-A', 'server:celery_app', 'worker', '-c 2 > log-worker.txt', '--loglevel=DEBUG']
CMD ["python3", "server.py"]
#ENTRYPOINT ["celery -A server:celery_app worker & python3 server.py;"]