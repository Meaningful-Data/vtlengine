from pathlib import Path
from time import time

from vtlengine.API import run

dev_name = 'BOP'
base_path = Path(__file__).parent / 'development' / 'data' / dev_name
input_dp = base_path / 'dataPoints' / 'input'
output_dp = base_path / 'dataPoints' / 'output'
input_ds = base_path / 'dataStructures' / 'input'
ext_routines = base_path / 'externalRoutines'
vds = base_path / 'valueDomains'
vtl = base_path / 'vtl' / f'{dev_name}.vtl'

if __name__ == '__main__':
    time_vector = []
    num_executions = 3
    for i in range(num_executions):
        start = time()
        run(
            script=vtl,
            data_structures=input_ds,
            datapoints=input_dp,
            value_domains=vds,
            output_folder=output_dp,
        )
        end = time()
        total_time = round(end - start, 2)
        time_vector.append(total_time)
        print(f'Execution ({i + 1}/{num_executions}): {total_time}s')
    print('-' * 30)
    print(f'Average time: {round(sum(time_vector) / num_executions, 2)}s')
    print(f'Min time: {min(time_vector)}s')
    print(f'Max time: {max(time_vector)}s')
    print(f'Total time: {round(sum(time_vector), 2)}s')
