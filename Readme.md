# Meta-Learning Base

## Installation
This project requires Python 3.6, Docker and docker-compose. It consists of three different python projects:
- meta-learning-base
- sklearn-components
- pynisher

```bash
sudo apt-get install libatlas-base-dev libblas3 liblapack3 liblapack-dev libblas-dev gfortran
sudo apt install python3-pip build-essential

Download other projects and install them
```bash
cd ..
git clone https://gitlab.usu-research.ml/research/automl/sklearn-components.git
sudo pip3 install -r sklearn-components/requirements.txt
pip3 install -e sklearn-components

git clone https://github.com/Ennosigaeon/pynisher.git
pip3 install -e pynisher
```

Install system-packages
```bash
sudo apt install libpq-dev
```

Clone meta-learning-base project and install requirements
```bash
git clone http://gitlab.usu-research.ml/research/automl/meta-learning-base.git
pip3 install -r meta-learning-base/requirements.txt
```

Configure virtual memory to prevent OOM killer
```bash
sudo vim /etc/sysctl.conf

vm.overcommit_memory=2
vm.overcommit_ratio=100

sysctl -p
```

Add limbo configuration to assets/
```bash
copy file to VM with scp then move it in assets/
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
or take a look at the [configuration](config.py). Example execution
```bash
mkdir data
mkdir logfile

screen
python3 cli.py worker --work-dir ./data --sql-config assets/sql.yaml --s3-config assets/s3.yaml --logfile ./logfiles/log1`
```

### Using external storage and database
You can either use the provided docker-compose file to use an external database and S3 storage
```bash
docker-compose up
```
or you can configure an existing db. If you want to use S3 storage, you will have to provide a google service account
file.



## Stopping a worker

Get the pid of the worker via `ps aux | grep cli.py` and terminate the worker with SIGUSR1. On ubuntu this equals
`kill -10 <PID>`. This will perform a graceful shutdown after the evaluation of the current algorithm is finished.