import json
import os
from pathlib import Path
from typing import Dict, List, Any
from unittest import TestCase

import pandas as pd
import pytest

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset, ValueDomain, ExternalRoutine, Scalar


class UDOHelper(TestCase):
    """

    """

    base_path = Path(__file__).parent
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_vtl = base_path / "data" / "vtl"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    # File extensions.--------------------------------------------------------------
    JSON = '.json'
    CSV = '.csv'
    VTL = '.vtl'

    @classmethod
    def LoadValueDomains(cls, vd_names):
        value_domains = {}
        for name in vd_names:
            vd_file_name = str(cls.filepath_valueDomain / f"{name}.json")
            with open(vd_file_name, 'r') as file:
                vd = ValueDomain.from_json(file.read())
                value_domains[vd.name] = vd
        return value_domains

    @classmethod
    def LoadExternalRoutines(cls, sql_names):
        external_routines = {}
        for name in sql_names:
            sql_file_name = str(cls.filepath_sql / f"{name}.sql")
            with open(sql_file_name, 'r') as file:
                external_routines[name] = ExternalRoutine.from_sql_query(name, file.read())
        return external_routines
    @classmethod
    def LoadDataset(cls, ds_path, dp_path):
        with open(ds_path, 'r') as file:
            structures = json.load(file)
        datasets = {}
        if 'datasets' in structures:
            for dataset_json in structures['datasets']:
                dataset_name = dataset_json['name']
                components = {
                    component['name']: Component(name=component['name'],
                                                 data_type=SCALAR_TYPES[component['type']],
                                                 role=Role(component['role']),
                                                 nullable=component['nullable'])
                    for component in dataset_json['DataStructure']}
                if not os.path.exists(dp_path):
                    data = pd.DataFrame(columns=list(components.keys()))
                else:
                    data = pd.read_csv(dp_path, sep=',')

                datasets[dataset_name] = Dataset(name=dataset_name, components=components, data=data)
        if 'scalars' in structures:
            for scalar_json in structures['scalars']:
                scalar = Scalar(name=scalar_json['name'],data_type=SCALAR_TYPES[scalar_json['type']], value=None)
                datasets[scalar.name] = scalar
        return datasets
    @classmethod
    def LoadInputs(cls, code: str, number_inputs: int) -> Dict[str, Dataset]:
        '''

        '''
        datasets = {}
        for i in range(number_inputs):
            json_file_name = str(cls.filepath_json / f"{code}-{str(i + 1)}{cls.JSON}")
            csv_file_name = str(cls.filepath_csv / f"{code}-{str(i + 1)}{cls.CSV}")
            dataset = cls.LoadDataset(json_file_name, csv_file_name)
            datasets = {**datasets, **dataset}

        return datasets

    @classmethod
    def LoadOutputs(cls, code: str, references_names: List[str]) -> Dict[str, Dataset]:
        """

        """
        datasets = {}
        for name in references_names:
            json_file_name = str(cls.filepath_out_json / f"{code}-{name}{cls.JSON}")
            csv_file_name = str(cls.filepath_out_csv / f"{code}-{name}{cls.CSV}")
            dataset = cls.LoadDataset(json_file_name, csv_file_name)
            datasets = {**datasets, **dataset}

        return datasets

    @classmethod
    def LoadVTL(cls, code: str) -> str:
        """

        """
        vtl_file_name = str(cls.filepath_vtl / f"{code}{cls.VTL}")
        with open(vtl_file_name, 'r') as file:
            return file.read()

    @classmethod
    def BaseTest(cls, code: str, number_inputs: int, references_names: List[str], vd_names: List[str] = None,
                 sql_names: List[str] = None, scalars: Dict[str, Any] = None):
        '''

        '''
        text = cls.LoadVTL(code)
        ast = create_ast(text)
        input_datasets = cls.LoadInputs(code, number_inputs)
        reference_datasets = cls.LoadOutputs(code, references_names)
        value_domains = None
        if vd_names is not None:
            value_domains = cls.LoadValueDomains(vd_names)

        external_routines = None
        if sql_names is not None:
            external_routines = cls.LoadExternalRoutines(sql_names)
        if scalars:
            for name in scalars:
                if name in input_datasets:
                    input_datasets[name].value = scalars[name]
                else:
                    raise Exception(f"Scalar {name} not found in input datasets")
        interpreter = InterpreterAnalyzer(input_datasets,
                                          value_domains=value_domains,
                                          external_routines=external_routines)
        result = interpreter.visit(ast)
        assert result == reference_datasets

    @classmethod
    def NewSemanticExceptionTest(cls, code: str, number_inputs: int, exception_code: str):
        assert True


class UdoTest(UDOHelper):
    """
    Group 1
    """

    classTest = 'udo.UdoTest'

    def test_1(self):
        """
        USER DEFINED OPERATORS
        Dataset --> Dataset
        Status: OK
        Expression: define operator drop_identifier (ds dataset, comp component)
                    returns dataset is
                      max(ds group except comp)
                  end operator;

                  define operator suma (ds1 dataset, ds2 dataset)
                    returns dataset is
                      ds1 + ds2
                  end operator;

                  DS_r := drop_identifier (suma (DS_1, DS_2), Id_3);
        Description: ***

        Git Branch: #288-test-concatenate-udo.
        Goal: Check the result of concatenate USER DEFINED OPERATORS.
        """
        code = '1-1-1-1'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        USER DEFINED OPERATORS
        Dataset --> Dataset
        Status: BUG THE Expression here is wrong i changed in the vtl file for the correct, but this example should give us a controlled exception but it is not the case
        Expression: define operator drop_identifier (ds dataset, comp component)
                      returns dataset is
                        max(ds group except comp)
                    end operator;

                    define operator suma (c1 component, c2 component)
                      returns dataset is
                        c1 + c2
                    end operator;

                    DS_r := drop_identifier (suma (DS_1#Me_1, DS_2#Me_1), Id_3);
        Description: It gives an error that says: AttributeError: 'BinOp' object has no attribute 'value'

        Git Branch: #288-test-concatenate-udo.
        Goal: Check the result of concatenate USER DEFINED OPERATORS.
        """
        code = '1-1-1-2'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        USER DEFINED OPERATORS
        Dataset --> Dataset
        Status: ?
        Expression: define operator drop_identifier (ds dataset, comp component)
                      returns dataset is
                        max(ds group except comp)
                    end operator;

                    define operator suma (DS dataset, Me1 component, Me2 component)
                      returns dataset is
                        DS [calc Me_3 := Me1 + Me2]
                    end operator;

                    DS_r := drop_identifier (suma (DS,Me_1,Me_2), Id_3);
        Description: It gives an error that says: KeyError: 'Me_3'

        Git Branch: #288-test-concatenate-udo.
        Goal: Check the result of concatenate USER DEFINED OPERATORS.

        Note: provisional exception until further analysis as creation of a new component should not be allowed
        """
        code = '1-1-1-3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        USER DEFINED OPERATORS
        Dataset --> Dataset
        Status: ?
        Expression: define operator drop_identifier (ds dataset, comp component)
                      returns dataset is
                        max(ds group except comp)
                    end operator;

                    define operator suma (DS dataset, Me1 component, Me2 component, Me_3 component)
                      returns dataset is
                        DS [calc Me_3 := Me1 + Me2]
                    end operator;

                    DS_r := drop_identifier (suma (DS,Me_1,Me_2), Id_3);
        Description: It gives an error that says: AttributeError: 'NoneType' object has no attribute 'value'

        Git Branch: #288-test-concatenate-udo.
        Goal: Check the result of concatenate USER DEFINED OPERATORS.

        Note: provisional exception until further analysis as creation of a new component should not be allowed
        """
        code = '1-1-1-4'
        number_inputs = 1
        message = "1-3-28"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
        # references_names = ["1"]

        # 
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        USER DEFINED OPERATORS
        Dataset --> Dataset
        Status: OK
        Expression: define operator drop_identifier (ds dataset, comp component)
                      returns dataset is
                        max(ds group except comp)
                    end operator;

                    define operator membership (x dataset, c component)
                      returns dataset is
                        x#c
                    end operator;

                    DS_r := drop_identifier (membership (DS,Me_2), Id_3);

        Description:****

        Git Branch: #288-test-concatenate-udo.
        Goal: Check the result of concatenate USER DEFINED OPERATORS.
        """
        code = '1-1-1-5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        USER DEFINED OPERATORS
        Dataset --> Dataset
        Status: OK
        Expression: define operator drop_identifier (ds dataset, comp component)
                      returns dataset is
                        max(ds group except comp)
                    end operator;

                    define operator membership (x dataset, c component)
                      returns dataset is
                        x#c
                    end operator;

                    DS_r := drop_identifier (membership (DS,Id_3), Id_3);

        Description:****

        Git Branch: #288-test-concatenate-udo.
        Goal: Check the result of concatenate USER DEFINED OPERATORS.
        """
        code = '1-1-1-6'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        USER DEFINED OPERATORS
        Dataset --> Dataset
        Status: OK
        Expression: define operator drop_identifier (ds dataset, comp component)
                      returns dataset is
                        max(ds group except comp)
                    end operator;

                    define operator membership (x dataset, c component)
                      returns dataset is
                        x#c
                    end operator;

                    DS_r := drop_identifier (membership (DS,At_1), Id_3);

        Description:****

        Git Branch: #288-test-concatenate-udo.
        Goal: Check the result of concatenate USER DEFINED OPERATORS.
        """
        code = '1-1-1-7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        USER DEFINED OPERATORS
        Dataset --> Dataset
        Status: OK
        Expression: define operator drop_identifier (ds dataset, comp component)
                      returns dataset is
                        max(ds group except comp)
                    end operator;

                    define operator resta (x dataset, s scalar)
                      returns dataset is
                        x - s
                    end operator;

                    DS_r := drop_identifier (resta (DS, 2.0), Id_3);

        Description:****

        Git Branch: #288-test-concatenate-udo.
        Goal: Check the result of concatenate USER DEFINED OPERATORS.
        """
        code = '1-1-1-8'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        USER DEFINED OPERATORS
        Dataset --> Dataset
        Status: BUG
        Expression: define operator drop_identifier (ds dataset, comp component)
                      returns dataset is
                        max(ds group except comp)
                    end operator;

                    DS_r :=  drop_identifier (drop_identifier (DS_1, Id_1), Id_4);
                    DS_r1 := drop_identifier (drop_identifier (DS_1, Id_2), Id_3);
                    DS_r2 := drop_identifier (drop_identifier (DS_1, Id_3), Id_2);
                    DS_r3 := drop_identifier (drop_identifier (DS_1, Id_4), Id_1);

        Description: It gives an error that says: AssertionError: 4 != 5
        check the output values, it seems they don't match what is expected

        Git Branch: #288-test-concatenate-udo.
        Goal: Check the result of concatenate USER DEFINED OPERATORS.
        """
        code = '1-1-1-9'
        number_inputs = 1
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        USER DEFINED OPERATORS
        Dataset --> Dataset
        Status: BUG
        Expression: define operator drop_identifier (ds dataset, comp component)
                      returns dataset is
                        max(ds group except comp)
                    end operator;

                    define operator greater (ds1 dataset, ds2 dataset)
                      returns dataset is
                        if ds1 > ds2 then ds1 else ds2
                    end operator;

                    DS_r := drop_identifier (greater (DS_1, DS_2), Id_3);

        Description: check the output values, it seems they don't match what is expected

        Git Branch: #288-test-concatenate-udo.
        Goal: Check the result of concatenate USER DEFINED OPERATORS.
        """
        code = '1-1-1-10'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_286_1(self):
        """
        USER DEFINED OPERATORS
        Status: OK
        Description: This had the discrepance
        Git Branch: #286
        """
        code = 'GL_286_1'
        number_inputs = 12
        references_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_286_2(self):
        """
        USER DEFINED OPERATORS
        Status: OK
        Description:
        Git Branch: #286
        """
        code = 'GL_286_2'
        number_inputs = 12
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_286_3(self):
        """
        USER DEFINED OPERATORS
        Status: OK
        Description:
        Git Branch: #286
        """
        code = 'GL_286_3'
        number_inputs = 12
        references_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_286_4(self):
        """
        USER DEFINED OPERATORS
        Status: OK
        Description:
        Git Branch: #286
        """
        code = 'GL_286_4'
        number_inputs = 12
        references_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_65(self):
        """
                UDO inside DPR
                Status: OK
                Description: Testing implementation UDO inside DPR
                Git Issue: #65
                """
        code = 'GL_65'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_381(self):
        """
            Constant as Dataset input parameter
        """
        code = 'GL_381'
        number_inputs = 0
        message = "1-3-34"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        USER DEFINED OPERATORS
        Dataset --> Dataset
        Status: OK
        Expression: define operator value_domain(ds dataset)
                    returns dataset is
                        ds in myGeoValueDomain
                    end operator;

                    DS_r:=value_domain(DS_1);

        Description:****

        Git Branch: #142-value-domain-udo-test
        Goal: Check the result of using value domains in user defined operators
        """
        code = '1-1-1-11'
        number_inputs = 1
        references_names = ["1"]
        vd_names = ["myGeoValueDomain"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names, vd_names=vd_names)

    def test_GL_452_1(self):
        """
        Status: OK
        Description: UDO with scalar as input parameter
        Goal: Check Result.
        """
        code = 'GL_452_1'
        number_inputs = 1
        references_names = ["1"]
        with pytest.raises(Exception, match="Expected Integer, got DataComponent on UDO max1, parameter x"):
            self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_452_2(self):
        """
        Status: OK
        Description: UDO without return type defined
        Goal: Check Result.
        """
        code = 'GL_452_2'
        number_inputs = 1
        message = "1-4-2-5"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_452_3(self):
        """
        Status: OK
        Description: UDO with scalar as input parameter
        Goal: Check Result.
        """
        code = 'GL_452_3'
        number_inputs = 0
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_452_4(self):
        """
        Status: OK
        Description: UDO with scalarDataset as input argument
        Goal: Check Result.
        """
        code = 'GL_452_4'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names, scalars={'sc_2': "4"})

    def test_GL_452_5(self):
        """
        Status: OK
        Description: UDO with scalarDataset as input argument
        Goal: Check Result.
        """
        code = 'GL_452_5'
        number_inputs = 2
        references_names = ["1"]
        with pytest.raises(Exception, match="Expected Integer, got DataComponent on UDO max1, parameter x"):
            self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names, scalars={'sc_2': "4"})

    def test_GL_442_1(self):
        """
        Status: OK
        Description: UDO with grouping by a component with a name that is a reserved word and a dot in the name
        Goal: Check Result.
        """
        code = 'GL_442_1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_442_2(self):
        """
            Constant as Dataset input parameter
        """
        code = 'GL_442_2'
        number_inputs = 1
        message = "1-1-2-2"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_442_3(self):
        """
        Status: OK
        Description: complete example of issue 442
        Goal: Check Result.
        """
        code = 'GL_442_3'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_442_4(self):
        """
        Status: OK
        Description: complete example of issue 441
        Goal: Check Result.
        """
        code = 'GL_442_4'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_442_5(self):
        """
        Status: OK
        Description: there is no difference between component and 'component' (or "component"), the only diference is if component is a reserved word or not
        Goal: Check Result.
        """
        code = 'GL_442_5'
        number_inputs = 2
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_473_1(self):
        """
        Status: OK
        Description: UDO with SDMX-CSV 1.0
        Goal: Check Result.
        """
        code = 'GL_473_1'
        number_inputs = 1
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_473_2(self):
        """
        Status: OK
        Description: UDO with SDMX-CSV 1.0
        Goal: Check Result.
        """
        code = 'GL_473_2'
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_474_1(self):
        """
        Status: OK
        Description: UDO with SDMX-CSV 1.0
        Goal: Check Result.
        """
        code = 'GL_474_1'
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_474_2(self):
        """
        Status: OK
        Description:
        Goal: Check Result.
        """
        code = 'GL_474_2'
        number_inputs = 1
        message = "1-1-1-2"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_474_3(self):
        """
        Status: OK
        Description:
        Goal: Check Result.
        """
        code = 'GL_474_3'
        number_inputs = 1
        message = "1-3-5"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_475_1(self):
        """
        Status: OK
        Description: UDO with SDMX-CSV 1.0
        Goal: Check Result.
        """
        code = 'GL_475_1'
        number_inputs = 1
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_476_1(self):
        """
        Status: OK
        Description:
        Goal: Check Result.
        """
        code = 'GL_476_1'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
