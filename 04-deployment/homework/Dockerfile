FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# do stuff here

RUN pip install -U pip

RUN pip install pipenv

COPY [ "starter.py", "./" ]

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT [ "python", "starter.py", "2023", "5" ]


