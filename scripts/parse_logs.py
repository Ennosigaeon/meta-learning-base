from os import listdir
from os.path import isfile, join
import re

from database import Database


def process_dir(base_dir, conn):
    logfiles = [join(base_dir, f) for f in listdir(base_dir) if isfile(join(base_dir, f))]

    for logfile in logfiles:
        with open(logfile, 'r') as f:
            lines = f.readlines()

        chunks = re.split(r'^.+ - INFO - worker - Starting new algorithm', ''.join(lines), flags=re.MULTILINE)[1:]

        algorithm_error_regex = re.compile(
            r'(Algorithm raised exception:)|(Algorithm violated)|(- core - Something went wrong.)')
        algorithm_id_regex = re.compile(r'Saved algorithm (\d+)')
        dataset_name_regex = re.compile(r'Creating dataset ([\w-]+)')
        duplicate_id_regex = re.compile(r'New dataset equals dataset (\d+)')

        for chunk in chunks:
            if algorithm_error_regex.search(chunk) is not None:
                # print('Skipping due to algorithm error')
                continue

            m = algorithm_id_regex.search(chunk)
            if m:
                algorithm_id = m.group(1)
            else:
                print('Failed to extract algorithm_id from """{}"""'.format(chunk))
                continue

            m = dataset_name_regex.search(chunk)
            if m:
                dataset_name = m.group(1)
            else:
                print('Failed to extract dataset_name from """{}"""'.format(chunk))
                continue

            m = duplicate_id_regex.search(chunk)
            if m:
                dataset = m.group(1)
            else:
                rs = conn.execute('SELECT id FROM datasets WHERE name = \'{}\''.format(dataset_name))
                if rs.rowcount > 1:
                    print('Found multiple datasets with name {}'.format(dataset_name))
                    continue
                elif rs.rowcount == 0:
                    print('Found no dataset with name {}'.format(dataset_name))
                    continue
                else:
                    dataset = next(rs)['id']

            print('Setting {} to output {}'.format(algorithm_id, dataset))
            conn.execute('''
                UPDATE algorithms SET
                output_dataset = {}
                WHERE id = {};
                '''.format(dataset, algorithm_id))

        print('\n\n\n')


if __name__ == '__main__':
    db = Database('postgres', 'postgres', 'postgres', 'usu4867!', '35.242.255.138', 5432)

    engine = db.engine
    with engine.connect() as conn:
        process_dir('../logfiles/mlb-9', conn)
        process_dir('../logfiles/mlb-10', conn)
