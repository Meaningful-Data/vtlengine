# VTL Engine

## Introduction

The VTL Engine is a Python library that allows you to create and run VTL scripts.
It is a Python based library around the [VTL Language](http://sdmx.org/?page_id=5096)
that allows you to validate and run VTL scripts.

## Installation

### Requirements

The VTL Engine requires Python 3.10 or higher.

### Install with pip
To install the VTL Engine on any Operating System, you can use pip:

```bash
pip install vtlengine
```

## Usage

The VTL Engine can be used to validate and run VTL scripts.
For more information, please refer to the API documentation.

### Semantic Analysis
The VTL Engine can be used to validate VTL scripts. Here is an example:

```python
from API import semantic_analysis
from pathlib import Path

script = Path("path/to/your/script.vtl")
datastructures = Path("path/to/your/datastructures/folder")
value_domains = [Path("path/to/your/value_domains/file1.json"), Path("path/to/your/value_domains/file2.json")]
external_routines = [Path("path/to/your/external_routines/file1.sql"), Path("path/to/your/external_routines/file2.sql")]

semantic_analysis(script=script, data_structures=datastructures, 
                  value_domains=value_domains, external_routines=external_routines)
```

### Run VTL Scripts

The VTL Engine can also be used to execute VTL scripts. Here is an example:

```python

from API import run
from pathlib import Path

script = Path("path/to/your/script.vtl")
datastructures = Path("path/to/your/datastructures/folder")
datapoints = Path("path/to/your/datapoints/folder")
output_folder = Path("path/to/your/output/folder")

value_domains = None
external_routines = None

run(script=script, data_structures=datastructures, datapoints=datapoints,
    value_domains=value_domains, external_routines=external_routines,
    output_path=output_folder, return_only_persistent=True
    )
```

The VTL engine will efficiently load the data points and data structures into memory and execute the script.
