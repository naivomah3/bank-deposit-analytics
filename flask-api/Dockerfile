FROM continuumio/miniconda3:latest
RUN pip install --upgrade pip
COPY . /usr/app/
EXPOSE 5000
WORKDIR /usr/app/
RUN pip install -r requirements.txt
CMD python app.py