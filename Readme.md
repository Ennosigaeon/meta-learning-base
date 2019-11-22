# Meta-Learning Base

## Prerequisites

This project requires Python 3.7, Docker and docker-compose

Install requirements
```bash
pip3 install -r requirements.txt
```

Start docker containers
```bash
docker-compose up
```

Pass configuration
```bash
python3 cli.py <COMMAND> --s3-config assets/s3.yaml --sql-config assets/sql.yaml
```