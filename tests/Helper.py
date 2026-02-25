import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest import TestCase

import pytest

from vtlengine.API import create_ast, run
from vtlengine.DataTypes import SCALAR_TYPES
from vtlengine.Exceptions import (
    RunTimeError,
    SemanticError,
    VTLEngineException,
    check_key,
)
from vtlengine.files.output import (
    TimePeriodRepresentation,
    format_time_period_external_representation,
)
from vtlengine.files.parser import load_datapoints
from vtlengine.Interpreter import InterpreterAnalyzer
from vtlengine.Model import (
    Component,
    Dataset,
    ExternalRoutine,
    Role,
    Role_keys,
    Scalar,
    ValueDomain,
)

# VTL_ENGINE_BACKEND can be "pandas" (default) or "duckdb"
VTL_ENGINE_BACKEND = os.environ.get("VTL_ENGINE_BACKEND", "pandas").lower()


def _use_duckdb_backend() -> bool:
    """Check if DuckDB backend should be used."""
    return VTL_ENGINE_BACKEND == "duckdb"


class TestHelper(TestCase):
    """ """

    # Path Selection.----------------------------------------------------------
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"

    JSON = ".json"
    CSV = ".csv"
    VTL = ".vtl"

    # Prefix for the input files (DS_).
    ds_input_prefix = ""

    @classmethod
    def LoadDataset(
        cls, ds_path, dp_path, only_semantic=False
    ) -> Dict[str, Union[Dataset, Scalar]]:
        with open(ds_path, "r") as file:
            structures = json.load(file)

        datasets = {}

        if "datasets" in structures:
            for dataset_json in structures["datasets"]:
                dataset_name = dataset_json["name"]
                components = {}

                for component in dataset_json["DataStructure"]:
                    check_key("data_type", SCALAR_TYPES.keys(), component["type"])
                    check_key("role", Role_keys, component["role"])
                    components[component["name"]] = Component(
                        name=component["name"],
                        data_type=SCALAR_TYPES[component["type"]],
                        role=Role(component["role"]),
                        nullable=component["nullable"],
                    )

                if only_semantic:
                    data = None
                else:
                    data = load_datapoints(components, dataset_name, Path(dp_path))

                datasets[dataset_name] = Dataset(
                    name=dataset_name, components=components, data=data
                )
        if "scalars" in structures:
            for scalar_json in structures["scalars"]:
                scalar_name = scalar_json["name"]
                scalar = Scalar(
                    name=scalar_name,
                    data_type=SCALAR_TYPES[scalar_json["type"]],
                    value=None,
                )
                datasets[scalar_name] = scalar
        return datasets

    @classmethod
    def LoadInputs(cls, code: str, number_inputs: int, only_semantic=False) -> Dict[str, Dataset]:
        """ """
        datasets = {}
        for i in range(number_inputs):
            json_file_name = str(
                cls.filepath_json / f"{code}-{cls.ds_input_prefix}{str(i + 1)}{cls.JSON}"
            )
            csv_file_name = str(
                cls.filepath_csv / f"{code}-{cls.ds_input_prefix}{str(i + 1)}{cls.CSV}"
            )
            new_datasets = cls.LoadDataset(json_file_name, csv_file_name, only_semantic)
            for x in new_datasets:
                if x in datasets:
                    raise Exception(f"Trying to redefine input datasets: {x}")
            datasets.update(new_datasets)

        return datasets

    @classmethod
    def LoadOutputs(
        cls, code: str, references_names: List[str], only_semantic=False
    ) -> Dict[str, Dataset]:
        """ """
        datasets = {}
        for name in references_names:
            json_file_name = str(cls.filepath_out_json / f"{code}-{name}{cls.JSON}")
            csv_file_name = str(cls.filepath_out_csv / f"{code}-{name}{cls.CSV}")
            new_datasets = cls.LoadDataset(json_file_name, csv_file_name, only_semantic)
            for dataset in new_datasets.values():
                dataset.ref_name = name
            datasets.update(new_datasets)

        return datasets

    @classmethod
    def LoadVTL(cls, code: str) -> str:
        """ """
        vtl_file_name = str(cls.filepath_VTL / f"{code}{cls.VTL}")
        with open(vtl_file_name, "r") as file:
            return file.read()

    @classmethod
    def BaseTest(
        cls,
        code: str,
        number_inputs: int,
        references_names: List[str],
        vd_names: List[str] = None,
        sql_names: List[str] = None,
        text: Optional[str] = None,
        scalars: Dict[str, Any] = None,
        only_semantic=False,
    ):
        warnings.filterwarnings("ignore", category=FutureWarning)
        if text is None:
            text = cls.LoadVTL(code)

        reference_datasets = cls.LoadOutputs(code, references_names, only_semantic)
        value_domains = None
        if vd_names is not None:
            value_domains = cls.LoadValueDomains(vd_names)

        external_routines = None
        if sql_names is not None:
            external_routines = cls.LoadExternalRoutines(sql_names)

        # Use DuckDB backend if configured
        if _use_duckdb_backend() and not only_semantic:
            result = cls._run_with_duckdb_backend(
                code=code,
                number_inputs=number_inputs,
                script=text,
                vd_names=vd_names,
                sql_names=sql_names,
                scalars=scalars,
            )
        else:
            # Original Pandas/Interpreter backend
            ast = create_ast(text)
            input_datasets = cls.LoadInputs(code, number_inputs, only_semantic)

            if scalars is not None:
                for scalar_name, scalar_value in scalars.items():
                    if scalar_name not in input_datasets:
                        raise Exception(f"Scalar {scalar_name} not found in the input datasets")
                    if not isinstance(input_datasets[scalar_name], Scalar):
                        raise Exception(f"{scalar_name} is a dataset")
                    input_datasets[scalar_name].value = scalar_value

            datasets = {k: v for k, v in input_datasets.items() if isinstance(v, Dataset)}
            scalars_obj = {k: v for k, v in input_datasets.items() if isinstance(v, Scalar)}

            interpreter = InterpreterAnalyzer(
                datasets=datasets,
                scalars=scalars_obj,
                value_domains=value_domains,
                external_routines=external_routines,
                only_semantic=only_semantic,
            )
            result = interpreter.visit(ast)

        for dataset in result.values():
            format_time_period_external_representation(
                dataset, TimePeriodRepresentation.SDMX_REPORTING
            )

        if len(result) != len(reference_datasets):
            diff_datasets = set(result.keys()) ^ set(reference_datasets.keys())
            raise Exception(
                f"Expected {len(reference_datasets)} datasets, got {len(result)}, difference: {diff_datasets}"
            )

        # cls._override_structures(code, result, reference_datasets)
        # cls._override_data(code, result, reference_datasets)
        assert result == reference_datasets

    @classmethod
    def _run_with_duckdb_backend(
        cls,
        code: str,
        number_inputs: int,
        script: str,
        vd_names: List[str] = None,
        sql_names: List[str] = None,
        scalars: Dict[str, Any] = None,
    ) -> Dict[str, Union[Dataset, Scalar]]:
        """
        Execute test using DuckDB backend.
        """
        # Collect data structure JSON files
        data_structures = []
        for i in range(number_inputs):
            json_file = cls.filepath_json / f"{code}-{cls.ds_input_prefix}{str(i + 1)}{cls.JSON}"
            data_structures.append(json_file)

        # Collect datapoint CSV paths
        datapoints = {}
        for i in range(number_inputs):
            json_file = cls.filepath_json / f"{code}-{cls.ds_input_prefix}{str(i + 1)}{cls.JSON}"
            csv_file = cls.filepath_csv / f"{code}-{cls.ds_input_prefix}{str(i + 1)}{cls.CSV}"
            # Load structure to get dataset names
            with open(json_file, "r") as f:
                structure = json.load(f)
            if "datasets" in structure:
                for ds in structure["datasets"]:
                    datapoints[ds["name"]] = csv_file
            # Scalars don't need datapoints

        # Load value domains if specified
        value_domains = None
        if vd_names is not None:
            value_domains = [cls.filepath_valueDomain / f"{name}.json" for name in vd_names]

        # Load external routines if specified
        external_routines = None
        if sql_names is not None:
            external_routines = cls.LoadExternalRoutines(sql_names)

        # Prepare scalar values
        scalar_values = None
        if scalars is not None:
            scalar_values = scalars

        return run(
            script=script,
            data_structures=data_structures,
            datapoints=datapoints,
            value_domains=value_domains,
            external_routines=external_routines,
            scalar_values=scalar_values,
            return_only_persistent=False,
            use_duckdb=True,
        )

    @classmethod
    def _override_structures(cls, code, result, reference_datasets):
        for dataset in result.values():
            ref_dataset = reference_datasets[dataset.name]
            param_name = ref_dataset.ref_name
            json_file_name = str(cls.filepath_out_json / f"{code}-{param_name}{cls.JSON}")
            with open(json_file_name, "w") as file:
                file.write(dataset.to_json_datastructure())

    @classmethod
    def _override_data(cls, code, result, reference_datasets):
        for dataset in result.values():
            if dataset.data is None:
                continue
            ref_dataset = reference_datasets[dataset.name]
            param_name = ref_dataset.ref_name
            csv_file_name = str(cls.filepath_out_csv / f"{code}-{param_name}{cls.CSV}")
            dataset.data.to_csv(csv_file_name, index=False, header=True)

    @classmethod
    def NewSemanticExceptionTest(
        cls,
        code: str,
        number_inputs: int,
        exception_code: str,
        vd_names: List[str] = None,
        sql_names: List[str] = None,
        text: Optional[str] = None,
        scalars: Dict[str, Any] = None,
    ):
        # Data Loading.--------------------------------------------------------
        warnings.filterwarnings("ignore", category=FutureWarning)
        if text is None:
            text = cls.LoadVTL(code)
        input_datasets = cls.LoadInputs(code=code, number_inputs=number_inputs)

        value_domains = None
        if vd_names is not None:
            value_domains = cls.LoadValueDomains(vd_names)

        external_routines = None
        if sql_names is not None:
            external_routines = cls.LoadExternalRoutines(sql_names)

        if scalars is not None:
            for scalar_name, scalar_value in scalars.items():
                if scalar_name not in input_datasets:
                    raise Exception(f"Scalar {scalar_name} not found in the input datasets")
                if not isinstance(input_datasets[scalar_name], Scalar):
                    raise Exception(f"{scalar_name} is a dataset")
                input_datasets[scalar_name].value = scalar_value

        datasets = {k: v for k, v in input_datasets.items() if isinstance(v, Dataset)}
        scalars_obj = {k: v for k, v in input_datasets.items() if isinstance(v, Scalar)}

        interpreter = InterpreterAnalyzer(
            datasets=datasets,
            scalars=scalars_obj,
            value_domains=value_domains,
            external_routines=external_routines,
        )
        with pytest.raises((SemanticError, RunTimeError)) as context:
            ast = create_ast(text)
            interpreter.visit(ast)

        result = exception_code == str(context.value.args[1])
        if result is False:
            print(f"\n{exception_code} != {context.value.args[1]}")
        assert result

    @classmethod
    def SemanticExceptionTest(
        cls,
        code: str,
        number_inputs: int,
        exception_code: str,
        vd_names: List[str] = None,
        sql_names: List[str] = None,
        text: Optional[str] = None,
        scalars: Dict[str, Any] = None,
    ):
        # Data Loading.--------------------------------------------------------
        warnings.filterwarnings("ignore", category=FutureWarning)
        if text is None:
            text = cls.LoadVTL(code)
        input_datasets = cls.LoadInputs(code=code, number_inputs=number_inputs)

        value_domains = None
        if vd_names is not None:
            value_domains = cls.LoadValueDomains(vd_names)

        external_routines = None
        if sql_names is not None:
            external_routines = cls.LoadExternalRoutines(sql_names)

        if scalars is not None:
            for scalar_name, scalar_value in scalars.items():
                if scalar_name not in input_datasets:
                    raise Exception(f"Scalar {scalar_name} not found in the input datasets")
                if not isinstance(input_datasets[scalar_name], Scalar):
                    raise Exception(f"{scalar_name} is a dataset")
                input_datasets[scalar_name].value = scalar_value

        interpreter = InterpreterAnalyzer(
            input_datasets,
            value_domains=value_domains,
            external_routines=external_routines,
        )
        with pytest.raises(SemanticError) as context:
            ast = create_ast(text)
            interpreter.visit(ast)

        result = exception_code == str(context.value.args[1])
        if result is False:
            print(f"\n{exception_code} != {context.value.args[1]}")
        assert result

    @classmethod
    def LoadValueDomains(cls, vd_names):
        value_domains = {}
        for name in vd_names:
            vd_file_name = str(cls.filepath_valueDomain / f"{name}.json")
            with open(vd_file_name, "r") as file:
                vd = ValueDomain.from_json(file.read())
                value_domains[vd.name] = vd
        return value_domains

    @classmethod
    def LoadExternalRoutines(cls, sql_names):
        external_routines = {}
        for name in sql_names:
            sql_file_name = str(cls.filepath_sql / f"{name}.sql")
            with open(sql_file_name, "r") as file:
                external_routines[name] = ExternalRoutine.from_sql_query(name, file.read())
        return external_routines

    @classmethod
    def DataLoadTest(cls, code: str, number_inputs: int, references_names: List[str] = None):
        # Data Loading.--------------------------------------------------------
        inputs = cls.LoadInputs(code=code, number_inputs=number_inputs)

        # Test Assertion.------------------------------------------------------
        if references_names:
            references = cls.LoadOutputs(code=code, references_names=references_names)
            assert inputs == references
        assert True

    @classmethod
    def DataLoadExceptionTest(
        cls,
        code: str,
        number_inputs: int,
        exception_message: Optional[str] = None,
        exception_code: Optional[str] = None,
    ):
        if exception_code is not None:
            with pytest.raises(VTLEngineException) as context:
                cls.LoadInputs(code=code, number_inputs=number_inputs)
        else:
            with pytest.raises(Exception, match=exception_message) as context:
                cls.LoadInputs(code=code, number_inputs=number_inputs)
        # Test Assertion.------------------------------------------------------

        if len(context.value.args) > 1 and exception_code is not None:
            assert exception_code == str(context.value.args[1])
        else:
            if exception_message is not None:
                assert exception_message in str(context.value.args[0])
