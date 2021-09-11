FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

WORKDIR /app
EXPOSE 8000

RUN pip install Flask
RUN pip install celery
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
ENTRYPOINT ["python3", "server.py"]
