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

## Usage

### Semantic Analysis
The VTL Engine can be used to semantically validate VTL scripts. Here is an example:

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

### Run VTL Scripts

The VTL Engine can also be used to execute VTL scripts. Here is an example:

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

The VTL engine will efficiently load the data points and data structures into memory and execute the script.

For more information on usage, please refer to the API documentation.
