# VTL Engine

## Introduction

The VTL Engine is a Python library that allows you to validate and run VTL scripts.
It is a Python-based library around the [VTL Language](http://sdmx.org/?page_id=5096).

## Installation

### Requirements

The VTL Engine requires Python 3.10 or higher.

### Install with pip
To install the VTL Engine on any Operating System, you can use pip:

```bash
pip install vtlengine
```

*Note: it is recommended to install the VTL Engine in a virtual environment.* 


## Usage

### Semantic Analysis
To perform the validation of a VTL script, please use the semantic_analysis function. 
Here is an example:

```python

from API import semantic_analysis
from pathlib import Path

base_path = Path(__file__).parent / "testSuite/API/data/"
script = base_path / Path("vtl/1.vtl")
datastructures = base_path / Path("DataStructure/input")
value_domains = base_path / Path("ValueDomain/VD_1.json")
external_routines = base_path / Path("sql/1.sql")

semantic_analysis(script=script, data_structures=datastructures, 
                  value_domains=value_domains, external_routines=external_routines)
```

The semantic analysis function will return a dictionary of the computed datasets and their structure.

### Run VTL Scripts

To execute a VTL script, please use the run function. Here is an example:

```python

from API import run
from pathlib import Path

base_path = Path(__file__).parent / "testSuite/API/data/"
script = base_path / Path("vtl/1.vtl")
datastructures = base_path / Path("DataStructure/input")
datapoints = base_path / Path("DataSet/input")
output_folder = base_path / Path("DataSet/output")

value_domains = None
external_routines = None

run(script=script, data_structures=datastructures, datapoints=datapoints,
    value_domains=value_domains, external_routines=external_routines,
    output_path=output_folder, return_only_persistent=True
    )
```
The VTL engine will load each datapoints file as being needed, reducing the memory footprint.
When the output parameter is set, the engine will write the result of the computation 
to the output folder, else it will include the data in the dictionary of the computed datasets.

For more information on usage, please refer to the [API documentation](https://docs.vtlengine.meaningfuldata.eu/api.html).
