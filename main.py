import json
from pathlib import Path
from time import time

from API import run

base_path = Path(__file__).parent / 'development' / 'data'
input_dp = base_path / 'dataPoints' / 'input'
output_dp = base_path / 'dataPoints' / 'output'
input_ds = base_path / 'dataStructures' / 'input'
ext_routines = base_path / 'externalRoutines'
vds = base_path / 'valueDomains'
vtl = base_path / 'vtl' / 'monthVal.vtl'

if __name__ == '__main__':
    start = time()
    run(
        script=vtl,
        data_structures=input_ds,
        datapoints=input_dp,
        value_domains=vds,
        output_path=output_dp,
    )
    end = time()
    print(f"Execution time: {round(end - start, 2)}s")
