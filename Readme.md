# Meta-Learning Base

## Installation
This project requires Python 3.6, Docker and docker-compose. It consists of three different python projects:
- meta-learning-base
- sklearn-components
- pynisher

Download other projects and install them
```bash
cd ..
git clone https://gitlab.usu-research.ml/research/automl/sklearn-components.git
git clone https://github.com/Ennosigaeon/pynisher.git

pip install -e sklearn-components
pip install -e pynisher
```

Install requirements
```bash
pip3 install -r meta-learning-base/requirements.txt
pip3 install -r sklearn-components/requirements.txt
```

## Running the applicaion

Pass configuration
```bash
python3 cli.py <COMMAND> --s3-config assets/s3.yaml --sql-config assets/sql.yaml
```
where <COMMAND> can be either `enter_data` or `worker`. For additional configuration options run
```bash
python3 cli.py <COMMAND> -h
```
or take a look at the [configuration](config.py).

### Using external storage and database
You can either use the provided docker-compose file to use an external database and S3 storage
```bash
docker-compose up
```
or you can configure an existing db. If you want to use S3 storage, you will have to provide a google service account
file.