"""Command Line Interfaced module."""

import logging

import argparse

from config import S3Config, DatasetConfig, LogConfig, SQLConfig, GenericConfig, WorkerConfig


def _get_core(args):
    from core import Core

    sql_conf = SQLConfig(args)
    s3_conf = S3Config(args)
    log_conf = LogConfig(args)
    generic_conf = GenericConfig(args)
    worker_conf = WorkerConfig(args)

    # Build params dictionary to pass to Core.
    core_args = sql_conf.to_dict()
    core_args.update(s3_conf.to_dict())
    core_args.update(log_conf.to_dict())
    core_args.update(generic_conf.to_dict())
    core_args.update(worker_conf.to_dict())

    return Core(**core_args)


def _work(args):
    """Creates a single worker."""
    core = _get_core(args)

    core.work(
        use_defaults=True,
        choose_randomly=False,
        wait=False
    )


def _enter_data(args):
    from data import load_data, load_openml

    core = _get_core(args)
    dataset_conf = DatasetConfig(args)

    if dataset_conf.openml is not None:
        df = load_openml(dataset_conf)
    elif dataset_conf.train_path is not None:
        df = load_data(dataset_conf.train_path)
    else:
        raise ValueError('Neither --openml nor --trainpath given.')

    class_column = dataset_conf.class_column

    dataset = core.add_dataset(df, class_column, depth=0, name=dataset_conf.name)

    return dataset.id


def _export_db(args):
    core = _get_core(args)
    core.export_db()
    print('Export db')


def _get_parser():
    logging_args = argparse.ArgumentParser(add_help=False)
    logging_args.add_argument('-v', '--verbose', action='count', default=0)
    logging_args.add_argument('-l', '--logfile')

    parser = argparse.ArgumentParser(description='Meta-Learning Base Command Line Interface',
                                     parents=[logging_args])

    subparsers = parser.add_subparsers(title='action', help='Action to perform')
    parser.set_defaults(action=None)

    # Common Arguments
    sql_args = SQLConfig.get_parser()
    s3_args = S3Config.get_parser()
    log_args = LogConfig.get_parser()
    dataset_args = DatasetConfig.get_parser()
    generic_args = GenericConfig.get_parser()

    # Enter Data Parser
    enter_data_parents = [
        logging_args,
        sql_args,
        s3_args,
        dataset_args,
        log_args,
        generic_args
    ]
    enter_data = subparsers.add_parser('enter_data', parents=enter_data_parents,
                                       help='Add a Dataset and trigger a Datarun on it.')
    enter_data.set_defaults(action=_enter_data)

    # Worker Args
    worker_args = argparse.ArgumentParser(add_help=False)
    worker_args.add_argument('--cloud-mode', action='store_true', default=False,
                             help='Whether to run this worker in cloud mode')
    worker_args.add_argument('--no-save', dest='save_files', action='store_false',
                             help="don't save models and metrics at all")
    computation_args = WorkerConfig.get_parser()

    # Worker
    worker_parents = [
        logging_args,
        worker_args,
        sql_args,
        s3_args,
        log_args,
        generic_args,
        computation_args
    ]
    worker = subparsers.add_parser('worker', parents=worker_parents,
                                   help='Start a single worker in foreground.')
    worker.set_defaults(action=_work)
    worker.add_argument('--datasets', help='Only train on datasets with these ids', nargs='+')
    worker.add_argument('--total-time', help='Number of seconds to run worker', type=int)

    export_db_parents = [
        logging_args,
        log_args,
        sql_args,
        generic_args
    ]
    export_db = subparsers.add_parser('export_db', parents=export_db_parents,
                                      help='Export all data sets and algorithms to a DataFrame')
    export_db.set_defaults(action=_export_db)

    return parser


def _logging_setup(verbosity=1, logfile=None):
    logger = logging.getLogger()
    log_level = (2 - verbosity) * 10
    fmt = '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    formatter = logging.Formatter(fmt)
    logger.setLevel(log_level)
    logger.propagate = False

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    logging.getLogger('botocore').setLevel(logging.CRITICAL)
    logging.getLogger('s3transfer').setLevel(logging.CRITICAL)
    logging.getLogger('urllib3').setLevel(logging.CRITICAL)
    logging.getLogger('mlb:worker').setLevel(logging.INFO)


def main():
    parser = _get_parser()
    args = parser.parse_args()

    _logging_setup(args.verbose, args.logfile)

    if not args.action:
        parser.print_help()
        parser.exit()

    args.action(args)


if __name__ == '__main__':
    main()
