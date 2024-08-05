import json
from pathlib import Path
from typing import Dict, List, Any
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset


class SemanticHelper(TestCase):
    """

    """

    base_path = Path(__file__).parent
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_vtl = base_path / "data" / "vtl"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    # File extensions.--------------------------------------------------------------
    JSON = '.json'
    CSV = '.csv'
    VTL = '.vtl'

    @classmethod
    def LoadDataset(cls, ds_path, dp_path):
        with open(ds_path, 'r') as file:
            structures = json.load(file)

        for dataset_json in structures['datasets']:
            dataset_name = dataset_json['name']
            components = {
                component['name']: Component(name=component['name'],
                                             data_type=SCALAR_TYPES[component['type']],
                                             role=Role(component['role']),
                                             nullable=component['nullable'])
                for component in dataset_json['DataStructure']}
            data = pd.read_csv(dp_path, sep=',')

            return Dataset(name=dataset_name, components=components, data=data)

    @classmethod
    def LoadInputs(cls, code: str, number_inputs: int) -> Dict[str, Dataset]:
        '''

        '''
        datasets = {}
        for i in range(number_inputs):
            json_file_name = str(cls.filepath_json / f"{code}-{str(i + 1)}{cls.JSON}")
            csv_file_name = str(cls.filepath_csv / f"{code}-{str(i + 1)}{cls.CSV}")
            dataset = cls.LoadDataset(json_file_name, csv_file_name)
            datasets[dataset.name] = dataset

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
            datasets[dataset.name] = dataset

        return datasets

    @classmethod
    def LoadVTL(cls, code: str) -> str:
        """

        """
        for i in code:
            if i == ['CC_', 'AI_', 'J_', 'Memb_' or 'Sc_']:
                vtl_file_name = str(cls.filepath_vtl / f"{code}{str(i + 1)}{cls.VTL}")
                with open(vtl_file_name, 'r') as file:
                    return file.read()

    @classmethod
    def BaseTest(cls, code: str, number_inputs: int, references_names: List[str]):
        '''

        '''

        text = cls.LoadVTL(code)
        ast = create_ast(text)
        input_datasets = cls.LoadInputs(code, number_inputs)
        reference_datasets = cls.LoadOutputs(code, references_names)
        interpreter = InterpreterAnalyzer(input_datasets)
        result = interpreter.visit(ast)
        assert result == reference_datasets

    @classmethod
    def NewSemanticExceptionTest(cls, code: str, number_inputs: int, exception_code: str):
        '''

        '''
        assert True


class ClauseClauseTests(SemanticHelper):
    """
    Group 1
    """

    classTest = 'Semantictests.ClauseClauseTests'

    def test_1(self):
        """
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := DS_1 [drop Me_2, Me_4][calc Me_2  := Me_1*2];
        Description: Calc after drop. Why is a bug?

        Git Branch:
        Goal: 
        """
        code = 'CC_1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [drop Me_2, Me_4][filter Me_2 = Me_1*2];
        Description: Filter after drop.

        Git Branch:
        Goal: .
        """
        code = 'CC_2'
        number_inputs = 1
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_3(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc me_3:= Me_1][keep me_3, Me_1][calc Me_4 := me_3*Me_1][keep me_3, Me_4];
        Description: Reusing calc and keep.

        Git Branch:
        Goal: .
        """
        code = 'CC_3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := (DS_1[calc Me_3:= Me_1][drop Me_2]*DS_2[filter Me_1>0])[drop Me_1][filter Me_3 < 0];
        Description: Two datasets with several clauses

        Git Branch:
        Goal: .
        """
        code = 'CC_4'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_3 := Me_2, Me_4 := Me_1][rename Me_3 to Me_3a, Me_4 to Me_4a];
        Description: One dataset with several clauses

        Git Branch:
        Goal: .
        """
        code = 'CC_5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_3 := Me_2, Me_4 := Me_1][drop Me_2][rename Me_3 to Me_3a, Me_4 to Me_4a][filter Me_4a < Me_1 + Me_3a];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_6'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_3 := Me_2][filter Me_3 = Id_2 or filter Me_2 > Me_1];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_3 := Me_2-Me_1, Me_4 := Me_1][keep Me_3, Me_2];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_8'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_3 := Me_2-Me_1, Me_4 := Me_1][keep Me_3, Id_2];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_9'
        number_inputs = 1
        error_code = "1-1-6-2"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_10(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_3 := Me_2-Me_1, Me_4 := Me_1][keep Me_3, Me_5];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_10'
        number_inputs = 1
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_11(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[sub Id_2 = "a"][calc identifier Id_3 := Id_1][calc Me_3 := Me_2=Me_1][keep Me_3];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_11'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[sub Id_1 = "a"][calc identifier Id_3 := Id_1][calc Me_3 := Me_2=Me_1][keep Me_3];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_12'
        number_inputs = 1
        error_code = "1-1-1-7"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_13(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_2 := Me_2, Me_3 := Me_2*Me_1][filter Me_3 < 0 and Me_2 >0][rename Me_3 to Me_3a];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_13'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[aggr Me_4 := sum(Me_1), Me_5 := min(Me_2) group by Id_1][calc Me_3 := Me_4/Me_5][drop Me_4];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_14'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_3 := Me_4/Me_5][aggr Me_4 := sum(Me_1), Me_5 := min(Me_2) group by Id_1][drop Me_4];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_15'
        number_inputs = 1
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_16(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_3 := Me_1/Me_2][aggr Me_4 := sum(Me_3), Me_5 := min(Me_2) group by Id_1][drop Me_4];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_16'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_3 := Me_1/Me_2][aggr Me_4 := sum(Me_3), Me_5 := count() group by Id_1][rename Me_4 to Me_2];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_17'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[aggr Me_4 := sum(Me_1), Me_5 := count() group by Id_1][rename Me_4 to Me_2];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_18'
        number_inputs = 1
        error_code = '1-1-1-2'

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_19(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[aggr Me_4 := sum(Me_1), Me_5 := count() group except Id_1][rename Me_4 to Me_2][calc Me_4 := Me_2+Me_5];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_19'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := min(DS_1 group except Id_1);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_20'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_21(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := sum(DS_1 group except Id_1);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_21'
        number_inputs = 1
        error_code = '1-1-1-2'

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_22(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := sum(DS_1[aggr Me_4 := sum(Me_1), Me_5 := count() group except Id_1][rename Me_5 to Me_c] group by Id_2);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_22'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_23(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := sum(DS_1[aggr Me_4 := sum(Me_1), Me_5 := count() group except Id_1][rename Me_5 to Me_c] group by Id_2)[calc Me_1 := Me_4, Me_a := Me_c*Me_4][keep Me_1, Me_a, Me_c];
        Description: Typical example that fails before refactor

        Git Branch:
        Goal: .
        """
        code = 'CC_23'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_24(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := sum(DS_1[aggr Me_4 := sum(Me_1), Me_5 := count() group except Id_1][rename Me_5 to Me_c])[calc Me_1 := Me_4, Me_a := Me_c*Me_4][keep Me_1, Me_a, Me_c];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_24'
        number_inputs = 1
        references_names = ["1"]
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_25(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := sum(DS_1[aggr Me_4 := sum(Me_5), Me_5 := count() group except Id_1] group by Id_2);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_25'
        number_inputs = 1
        error_code = '1-1-6-7'

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_26(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := sum(DS_1[aggr Me_4 := sum(Me_1), Me_5 := count() group except Id_1, Id_2] group by Id_3, Id_4);
        Description: Typical example that fails before refactor

        Git Branch:
        Goal: .
        """
        code = 'CC_26'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_27(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[aggr Me_4 := sum(Me_1), Me_5 := min(Me_4) group except Id_1, Id_2] ;
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_27'
        number_inputs = 1
        error_code = '1-1-6-7'

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_28(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [drop Me_4][filter Me_2 = Me_4*2];
        Description: Filter after drop.

        Git Branch:
        Goal: .
        """
        code = 'CC_28'
        number_inputs = 1
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_29(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [drop Me_2, Me_4][filter Me_2 = Me_1*2];
        Description: Filter after drop.

        Git Branch:
        Goal: .
        """
        code = 'CC_29'
        number_inputs = 1
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_30(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [unpivot Id_2a, Me_1];
        Description: Me_1 integer->string

        Git Branch:
        Goal: .
        """
        code = 'CC_30'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_31(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [ unpivot Id_2, Me_1];
        Description: Id_2 is in DS_1.

        Git Branch:
        Goal: .
        """
        code = 'CC_31'
        number_inputs = 1
        error_code = "1-1-6-2"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_32(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [ calc Me_4 := Me_1, Me_5 := Me_4];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_32'
        number_inputs = 1
        error_code = "1-1-6-7"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_33(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [ calc Me_4 := Me_1, Me_5 := Id_2 + Id_3];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_33'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_34(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_3 :=d1#Me_1 + d2#Me_11, Me_4 := Me_3 * 2.5);
        Description:Me_3 is overwrite but used too in the calc, but we use old Me_3 not the new Me_3

        Git Branch:
        Goal: .
        """
        code = 'CC_34'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_35(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join ( AMOUNTS [ sub measure_ = "O" ] [ rename OBS_VALUE to O ] [ drop OBS_STATUS ] as A ,  AMOUNTS [ sub measure_ = "V" ] [ rename OBS_VALUE to V ] [ drop OBS_STATUS ] as B ) ;
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_35'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_36(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join ( AMOUNTS [ sub measure_ = "O" ] [ rename OBS_VALUE to O ] [ drop OBS_STATUS ] as A ,  AMOUNTS [ sub measure_ = "V" ] [ rename OBS_VALUE to V ] [ drop OBS_STATUS ] as B ) ;
                    DS_r1 := DS_r[ sub V = "O" ];
        Description: A sub with measure

        Git Branch:
        Goal: .
        """
        code = 'CC_36'
        number_inputs = 1
        error_code = "1-1-6-10"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_37(self):
        """
        Dataset --> Dataset
        Status: OK, HACER UNO CON Id_1 tb que debe dar error
        Expression: define operator Test173 (x dataset, y component, z component)
                        returns dataset is
                        x [ unpivot y, z ]
                    end operator;
                    DS_r := Test173(DS_1, Id_2, Me_1);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_37'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_38(self):
        """
        Dataset --> Dataset
        Status: OK, HACER UNO CON me_1 tb que debe dar error y con paraemtros y argumentos cambiados
        Expression: define operator Test171 (x dataset, y component, z component, Me_2 component, At_2 component)
                        returns dataset is
                        x [ rename y to Me_2, z to At_2]
                    end operator;
                    DS_r := Test171(DS_1, Me_1, At_1,Me_2, At_2);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_38'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_39(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [drop Me_2][calc Me_2  := Me_1*2];
        Description: GL_264

        Git Branch:
        Goal: .
        """
        code = 'CC_39'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_40(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_3 := Me_2-Me_1, Me_4 := Me_1][keep Me_3, Me_5];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'CC_40'
        number_inputs = 2
        error_code = "1-1-1-15"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_41(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [filter Me_1 >= 1.0 and Me_2 < 3][calc Me_3 := Me_1 * Me_2, Me_4 := Me_1 / Me_2];
        Description: calc after filter. Me_1 is a Number and Me_2 is an Integer

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_41'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_42(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [filter Me_1 = true and Me_2 = false][calc Me_3 := Me_1 and Me_2, Me_4 := Me_3 and false];
        Description: "Cannot use component {comp_name} as it was generated in another calc expression."

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_42'
        number_inputs = 1
        error_code = "1-1-6-7"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_43(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [filter Me_1 >= 1.0 and Me_2 < 3][calc identifier Id_1 := Id_3];
        Description: "At op {op}: You could not recalculate the identifier {name}."

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_43'
        number_inputs = 1
        error_code = "1-1-12-1"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_44(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [drop Me_2, Me_4][calc Me_2 := Me_1, Me_3 := Me_1][keep Me_2, Me_3];
        Description:

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_44'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_45(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [drop Me_2, Me_4][rename Me_1 to Medate_1, Me_3 to Metp_3];
        Description: 

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_45'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_46(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc me_3:= Me_1][keep me_3, Me_1][calc Me_4 := me_3][keep me_3, Me_4];
        Description: 

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_46'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_47(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_3 := Me_2, Me_4 := Me_1][rename Me_3 to Me_3a, Me_4 to Me_4a];
        Description: 

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_47'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_48(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[sub Id_2 = "a"][calc identifier Id_3 := Id_1][calc Me_3 := Me_2=Me_1][keep Me_3];
        Description: "At op {op}: Invalid data type {type} for Component {name}." 
        Note: Me_1 is a Time_Period and Me_2 is a Date

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_48'
        number_inputs = 1
        error_code = "1-1-1-7"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_49(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[sub Id_2 = "a"][calc identifier Id_3 := Id_1][calc Me_3 := Me_2=Me_1][keep Me_3];
        Description: Me_1 and Me_2 are Date

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_49'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_50(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[aggr Me_3 := sum(Me_1), Me_4 := min(Me_2) group by Id_1, Id_2]
                                [rename Me_3 to Me_3A, Me_4 to Me_4A];
        Description: Me_1 is Number and Me_2 is an Integer

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_50'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_51(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [calc Me_4 := Me_1, Me_5 := Id_2 + Id_3][keep Me_3, Me_6]
                                 [rename Me_3 to Medate_3, Me_6 to Metp_6];
        Description: 

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_51'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_52(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [filter Id_1 = Id_2] [aggr Me_1:= sum( Me_1 ) group by Id_1, Id_2] [rename Id_1 to Id_1A, Me_1 to Me_1A];
        Description: 

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_52'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_53(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r:= DS_1 [calc Me_1:= Me_1 * 3.0, Me_2:= Me_2 * 2.0][keep Me_1, Me_2, Me_3, Me_4]
                                 [aggr Me_5 := sum(Me_1), Me_6 := min(Me_2) group except Id_1];
        Description: 

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_53'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_54(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := (DS_1[sub Id_2 = "A" ] + DS_1 [ sub Id_2 = "XX"])[rename Id_1 to Id_1A, Id_3 to Id_3A];
        Description: 

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_54'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_55(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [filter Me_1 = true and Me_2 = false][calc Me_3 := Me_1 and Me_2]
                                 [rename Me_1 to MeB_1, Me_2 to MeB_2, Me_3 to MeB_3];
        Description: 

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_55'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_56(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [filter Me_1 = true and Me_2 = false][calc Me_3 := Me_1 and Me_2]
                    [rename Me_1 to MeB_1, Me_2 to MeB_2, Me_3 to MeB_3][aggr Me_4 := min (MeB_1) group by Id_1];
        Description: 

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_56'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_57(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [filter Me_1 = true and Me_2 = false][calc Me_3 := Me_1 and Me_2]
                    [rename Me_1 to MeB_1, Me_2 to MeB_2, Me_3 to MeB_3][aggr Me_4 := max (MeB_1) group except Id_1];
        Description: 

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'CC_57'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_58(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [drop Me_2, Me_4][calc Me_2 := Me_1*2];
        Description: 

        Git Branch:
        Goal: 
        """
        code = 'CC_58'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class MembershipTests(SemanticHelper):
    """
    Group
    """

    classTest = 'Semantictests.MembershipTests'

    def test_1(self):
        """
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := DS_1#Id_1;
        Description: Membership#Identifier(Time, Date, Time Period).

        Git Branch:
        Goal: .
        """
        code = 'Memb_1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_3 := Me_1][rename DS_1#Me_1 to Me_1a];
        Description: Filter after drop.

        Git Branch:
        Goal: .
        """
        code = 'Memb_2'
        number_inputs = 1
        error_code = "1-1-6-6"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_5(self):
        """
        USER DEFINED OPERATORS
        Dataset --> Dataset
        Status:
        Expression: define operator drop_identifier (ds dataset, comp component)
                      returns dataset is
                        max(ds group except comp)
                    end operator;

                    define operator suma (c1 component, c2 component)
                      returns dataset is
                        c1 + c2
                    end operator;

                    DS_r := drop_identifier (suma (DS_1#Me_1, DS_2#Me_1), Id_3);
        Description: The test with the right expression is in udo tests, but i want to manage the exception here

        Git Branch: 
        Goal: Check the result of concatenate USER DEFINED OPERATORS.
        """
        code = 'Memb_5'
        number_inputs = 2
        error_code = "1-4-1-1"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_7(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1#Id_1;
        Description: Membership#Identifier(Time).

        Git Branch: 388-clause-clause-tests
        Goal: 
        """
        code = 'Memb_7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1#Id_1;
        Description: Membership#Identifier(Date).

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'Memb_8'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1#Id_1;
        Description: Membership#Identifier(Time_Period).

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'Memb_9'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1#At_1;
        Description: Membership#Attribute(Time).

        Git Branch: 388-clause-clause-tests
        Goal:
        """
        code = 'Memb_10'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class AliasTests(SemanticHelper):
    """
    Group
    """

    classTest = 'Semantictests.AliasTests'

    def test_1(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[rename Me_1 to Me_1a, Me_2 to Me_2a][rename Me_2a to Me_2];
        Description: two rename involved.

        Git Branch:
        Goal: .
        """
        code = 'Al_1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := DS_1[rename Me_1 to Me_1a, Me_2 to Me_2a][rename Me_2a to Me_2][rename Me_1a to Me_2a];
        Description: two rename involved.

        Git Branch:
        Goal: .
        """
        # error_code = "1-3-1"
        code = 'Al_2'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[rename Me_2 to Me_2a][drop Me_1][rename Me_2a to Me_1];
        Description: two rename involved.

        Git Branch:
        Goal: .
        """
        code = 'Al_3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_3 := Me_2][rename Me_3 to Me_3a];
        Description: calc and rename.

        Git Branch:
        Goal: .
        """
        code = 'Al_4'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := left_join ( DS_1 as d1, DS_2 as d2
                                filter d1#Me_1 + d2#Me_1 <10
                                calc Me_1 := d1#Me_1 + d2#Me_3
                                keep Me_1
                                rename Me_1 to Me_1a);
        Description:

        Git Branch:
        Goal: .
        this following example works but shouldnt
        DS_r := left_join ( DS_1 as d1, DS_2 as d2
            filter d1#Me_1 + d2#Me_1 <10
            calc Me_1 := d1#Me_1 + d2#Me_3
            keep d1#Me_1
            rename Me_1 to Me_1a);
        """
        code = 'Al_5'
        number_inputs = 2
        # joins
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join ( DS_1 as d1, DS_2 as d2
                                        keep Me_1
                                        rename Me_1 to d1);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Al_6'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1[calc Me_3 := Me_2][rename Me_1 to Me_3a];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Al_9'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_3 := Me_2][rename Me_5 to Me_3a];
        Description: Filter after drop.

        Git Branch:
        Goal: .
        """
        code = 'Al_10'
        number_inputs = 1
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_11(self):
        """
        Dataset --> Dataset
        Status: ok This should be a bug RM 7081
        Expression: DS_r := DS_1[calc Me_3 := Me_2][rename Me_1 to Me_3];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Al_11'
        number_inputs = 1
        error_code = '1-1-6-8'

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_12(self):
        """
        Dataset --> Dataset
        Status: ok This should be a bug RM 7081
        Expression: DS_r := DS_1[calc Me_3 := Me_1][rename Me_1 to Me_2];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Al_12'
        number_inputs = 1
        error_code = '1-1-6-8'

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_13(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_3 := Me_2][drop Me_2][rename Me_1 to Me_2];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Al_13'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        Dataset --> Dataset
        Status: ok This should be a bug RM 7081
        Expression: DS_r := DS_1[calc Me_3 := Me_2*Me_1, Me_4:= Me_3];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Al_14'
        number_inputs = 1
        error_code = '1-1-6-7'

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_15(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc identifier Id_3 := nvl(Me_2*Me_1, 0)][drop Id_3];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Al_15'
        number_inputs = 1
        error_code = '1-1-6-2'

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_16(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[rename Me_1 to Me_1a][calc Me_1 := Me_1a*Me_2][drop Me_1a];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Al_16'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[rename Me_1 to Me_1a][calc Me_1a := Me_2][drop Me_2];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Al_17'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[rename Me_1 to Me_1a][calc Me_2a := Me_1*Me_2];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Al_18'
        number_inputs = 1
        error_code = '1-3-16'

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_19(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[rename Me_1 to Me_1a][calc Id_2 := Me_1a*Me_2];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Al_19'
        number_inputs = 1
        error_code = '1-1-12-1'

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_21(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1[calc Me_3 := Me_1][rename Me_1 to Me_1a, Me_1 to Me_2a];
        Description: Filter after drop.

        Git Branch:
        Goal: .
        """
        code = 'Al_21'
        number_inputs = 1
        error_code = "1-1-6-9"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_22(self):
        """
        Dataset --> Dataset
        Status: ok This should be a bug RM 7081
        Expression: DS_r := DS_1[calc Me_3 := Me_1][rename Me_1 to Me_1a, Me_2 to Me_1a];
        Description: Filter after drop.

        Git Branch:
        Goal: .
        """
        code = 'Al_22'
        number_inputs = 1
        error_code = "1-3-1"  # "1-1-6-8"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_23(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := left_join ( DS_1 as d1, DS_2 as d2
            filter d1#Me_1 + d2#Me_1 <10
            calc Me_1 := d1#Me_1 + d1#Me_3
            keep Me_1
            rename Me_1 to Me_1a);
        Description:
        Git Branch:
        Goal: .
        """
        code = 'Al_23'
        number_inputs = 2
        error_code = "1-1-13-17"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)


class JoinTests(SemanticHelper):
    """
    """
    classTest = 'Semantictests.JoinTests'

    def test_1(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join ( DS_1 as d1, DS_2 as d2 keep d2#Me_1, Me_1A);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'J_1'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join ( DS_1 as d1, DS_2 as d2 keep d1#Me_1, Me_1A);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'J_2'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1 as d1, DS_2 as d2 keep d2#Me_1, Me_1A)[calc Me_3 := Me_1A*Me_1];
        Description: This fails actually in 0.11.x

        Git Branch:
        Goal: .
        """
        code = 'J_3'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join(DS_1 as d1, DS_2 as d2 calc Me_3 := Me_2*d2#Me_1 keep d1#Me_1, Me_1A, Me_3);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'J_4'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join ( DS_1 as d1, DS_2 as d2 keep d2#Me_1, Me_1A)[calc Me_3 := Me_2*Me_1];
        Description: This should be a fail because Me_2 not in dataset result from join

        Git Branch:
        Goal: .
        """
        code = 'J_5'
        number_inputs = 2
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_6(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join ( DS_1 as d1, DS_2 as d2 keep Me_1, Me_1A)[calc Me_3 := Me_2*Me_1];
        Description: Me_1 is ambiguous inside keep inside join

        Git Branch:
        Goal: .
        """
        code = 'J_6'
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_7(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join(DS_1[filter Id_2<>"aa"] as d1, DS_2 as d2 filter Id_4 > 0 calc Me_3 := Me_2*d2#Me_1 keep d1#Me_1, Me_1A, Me_3)[rename Me_3 to Me_3a];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'J_7'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join(DS_1 as d1, DS_2 as d2 calc Me_3 := Me_2*d2#Me_1, Me_4 := Id_4*d2#Me_1 keep d1#Me_1, Me_1A, Me_3)[calc Me_5 := Me_3*Me_4];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'J_8'
        number_inputs = 2
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_9(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join(DS_1[filter Id_2<>"aa"] as d1, DS_2 as d2 filter Id_4 > 0 calc Me_3 := Me_2*d2#Me_1, Me_4 := Id_4*d2#Me_1 keep d1#Me_1, Me_1A, Me_3, Me_4)[rename Me_3 to Me_3a][calc Me_5 := Me_3a*Me_4][drop Me_3a,Me_4];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'J_9'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join(DS_1[filter Id_2<>"aa"] as d1, DS_2 as d2 filter Me_1 > 0 calc Me_3 := Id_4*d2#Me_1 keep d1#Me_1, Me_1A, Me_3)[rename Me_3 to Me_3a];
        Description: filter with ambiguities

        Git Branch:
        Goal: .
        """
        code = 'J_10'
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_11(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join ( DS_1, DS_2 rename DS_1#Me_2 to Me_4, DS_2#Me_1A to Me_1B);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'J_11'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join ( DS_1, DS_2 rename DS_1#Me_2 to Me_4, Me_1A to Me_1B);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'J_12'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join ( DS_1, DS_2 rename DS_1#Me_2 to Me_4, DS_1#Me_1A to Me_1B);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'J_13'
        number_inputs = 2
        error_code = "1-1-13-17"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_14(self):
        """
        Dataset --> Dataset
        Status:
        Expression: TEST REPETIDO PERO QUIERO TENERLO AQUÍ
        Description: clauses in dataset used in join

        Git Branch:
        Goal: .
        """
        code = 'J_14'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join ( DS_1 as d1, DS_2[keep Me_1A, Me_2] as d2, DS_2[drop Me_1A, Me_2] as d3 keep d2#Me_2);
        Description: clauses in datasets used in join

        Git Branch:
        Goal: .
        """
        code = 'J_15'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_16(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join(DS_1[filter Id_2<>"aa"][rename Me_1 to Me_1New] as d1, DS_2 filter Me_1 > 0 calc Me_3 := Id_4*Me_1 keep d1#Me_1New, Me_1A, Me_3)[rename Me_3 to Me_3a];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'J_16'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join(DS_1[filter Id_2<>"aa"][rename Me_1 to Me_1New] as d1, DS_2 filter Me_1 > 0 calc Me_3 := Id_4*Me_1 keep DS_2#Me_1, Me_1A, Me_3)[rename Me_3 to Me_3a];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'J_17'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join(DS_1[filter Id_2<>"aa"][rename Me_1 to Me_1New] as d1, DS_2 filter Me_1 > 0 calc Me_3 := Id_4*Me_1 keep Me_1, Me_1A, Me_3)[rename Me_3 to Me_3a];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'J_18'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join(DS_1 as d1, DS_2 as d2 calc Me_3 := Me_2*d2#Me_1 keep DS_1#Me_1, Me_1A, Me_3);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'J_19'
        number_inputs = 2
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_20(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := inner_join (drop_identifier(DS_1, Id_3) as d1, drop_identifier(DS_2, Id_3) as d2 filter Me_2a < 0 calc identifier Id_4 :=nvl(d2#Me_1 + Me_2,0) keep d1#Me_1)[rename Me_1 to Me_2b];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'J_20'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class AggregateTests(SemanticHelper):
    """
    """
    classTest = 'Semantictests.AggregateTests'

    def test_1(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := min(DS_1 group by DT_RFRNC, PRSPCTV_ID);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'J_1'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := avg(DS_1 group except Id_1);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Ag_2'
        number_inputs = 1
        error_code = "1-1-1-2"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)


class ScalarTests(SemanticHelper):
    """
    """
    classTest = 'Semantictests.ScalarTests'

    def test_1(self):
        """
        Dataset --> Dataset
        Status:
        Expression: 
        Description: Error when redefine datasets

        Git Branch:
        Goal: .
        """
        code = 'Sc_1'
        number_inputs = 1
        error_code = "1-3-3"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_2(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Sc_2'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1*(sc_1+sc_2) ;
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Sc_3'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := sc_1*(sc_1+sc_2);
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Sc_4'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r1 := sc_1+sc_2 ;
                    DS_r2 := DS_r3 + sc_1+sc_2 ;
                    DS_r3 := DS_r2 + sc_1+sc_2 ;
        Description: cyclic graph

        Git Branch:
        Goal: .
        """
        code = 'Sc_5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r1 := sc_1 + DS_1;
        Description: Scalar overwritten a dataset

        Git Branch:
        Goal: .
        VtlEngine.Exceptions.exceptions.VTLEngineException: Trying to redefine input datasets. ['DS_1'].
        """
        code = 'Sc_6'
        number_inputs = 2
        error_code = "0-1-0-1"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_7(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1/sc_1 ;
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Sc_7'
        number_inputs = 1  # number of files
        error_code = "1-1-1-5"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_8(self):
        """
        Dataset --> Dataset
        Status:
        Expression: inner_join ;
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Sc_8'
        number_inputs = 1  # number of files
        error_code = "1-1-13-10"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_9(self):
        """
        Dataset --> Dataset
        Status:
        Expression: cross_join ;
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Sc_9'
        number_inputs = 1  # number of files
        error_code = "1-1-13-10"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_10(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 || sc_1 ;
        Description: String operation, string and number

        Git Branch:
        Goal: .
        """
        code = 'Sc_10'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := sc2 || sc_1 ;
        Description: String operation, string and number

        Git Branch:
        Goal: .
        """
        code = 'Sc_11'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1[calc Me_2:= Me_1 || sc_1];
        Description:Study this case

        Git Branch:
        Goal: .
        """
        code = 'Sc_12'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 <> sc_1;
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Sc_13'
        number_inputs = 1
        error_code = "1-1-1-2"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_14(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_1 := DS_1[calc identifier Id_3 := Me_1 <> sc_2];
        Description:

        Git Branch:
        Goal: .
        """
        code = 'Sc_14'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_1 := DS_1[calc identifier Id_3 := Me_1 <> sc_2];
        Description: sc_2 is also a component in DS_1

        Git Branch:
        Goal: .
        """
        code = 'Sc_15'
        number_inputs = 2
        error_code = "1-1-6-11"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_16(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := DS_1 [ aggr Me_2 := max ( Me_1 ) , Me_3 := min ( Me_1 ) group by Id_1 ]
                                 [calc Me_4 := Me_2 * sc_1, Me_5 := Me_3 + sc_2];
        Description:

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_16'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r1 := DS_1 and sc_1;
                    DS_r2 := DS_1 and sc_2;
                    DS_r := DS_r1 and DS_r2;
        Description:

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_17'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r1 := DS_1 or sc_1;
                    DS_r2 := DS_1 or sc_2;
                    DS_r := DS_r1 or DS_r2;
        Description:

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_18'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r:= DS_1[calc sc_2:= sc_1 or true];
        Description: 

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_19'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r:= DS_1[calc sc_2:= sc_1 xor true];
        Description:

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_20'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_21(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r:= DS_1[calc sc_2:= not sc_1];
        Description:

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_21'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_22(self):
        """
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := count (sc_1 group by Id_1);
        Description: 'Component Id_1 not found. Please check transformation with output dataset DS_r'
        ***The exception given by the engine is not the most appropriate, it should be another one
        related to the aggregate operators

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_22'
        number_inputs = 1
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_23(self):
        """
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := max (sc_1 group except Id_2);
        Description: 'Component Id_2 not found. Please check transformation with output dataset DS_r'
        ***The exception given by the engine is not the most appropriate, it should be another one
        related to the aggregate operators

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_23'
        number_inputs = 1
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_24(self):
        """
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := stddev_samp(sc_1 group by Id_2);
        Description: 'Component Id_2 not found. Please check transformation with output dataset DS_r'
        ***The exception given by the engine is not the most appropriate, it should be another one
        related to the aggregate operators

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_24'
        number_inputs = 1
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_25(self):
        """
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := last_value (sc_1 over (partition by Id_1 order by Id_2 data points between 1
                    preceding and 1 following));
        Description: 'Component Id_2 not found. Please check transformation with output dataset DS_r'
        ***The exception given by the engine is not the most appropriate, it should be another one
        related to the analytic operators

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_25'
        number_inputs = 1
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_26(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := if sc_1 = 0 then DS_1 else sc_2;
        Description: 'At op if: Then clause DS_1 and else clause sc_2, both must be Scalars. Please
                      check transformation with output dataset DS_r'

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_26'
        number_inputs = 2
        error_code = "1-1-9-3"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_27(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := if DS_1 = 0 then sc_1 else sc_2;
        Description: 'At op if: then clause sc_1 and else clause sc_2, both must be Datasets
        or at least one of them a Scalar. Please check transformation with output dataset DS_r'

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_27'
        number_inputs = 2
        error_code = "1-1-9-12"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_28(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := if DS_1 = 0 then DS_1 else sc_1;
        Description: 
        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_28'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_29(self):
        """
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := if sc_1 = 0 then sc_1 else sc_2;
        Description: AttributeError: 'NoneType' object has no attribute 'scalarType'

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_29'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_30(self):
        """
        Dataset --> Dataset
        Status:
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(sc_1, accountingEntry rule ACCOUNTING_ENTRY dataset);
        Description: Component ACCOUNTING_ENTRY not found. Please check transformation with output dataset DS_r'

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_30'
        number_inputs = 1
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_31(self):
        """
        Dataset --> Dataset
        Status:
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY non_zero dataset all_measures)
                            [aggr OBS_VALUE_N := sum (OBS_VALUE) group by ACCOUNTING_ENTRY][calc OBS_VALUE_N2 := OBS_VALUE_N + sc_1*sc_2];
        Description:

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_31'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_32(self):
        """
        Dataset --> Dataset
        Status:
        Expression: define datapoint ruleset signValidation (variable ACCOUNTING_ENTRY as AE, INT_ACC_ITEM as IAI,
        FUNCTIONAL_CAT as FC, INSTR_ASSET as IA, OBS_VALUE as O) is
        sign1c: when AE = "C" and IAI = "G" then O > 0 errorcode "sign1c" errorlevel 1;
        sign2c: when AE = "C" and IAI = "GA" then O > 0 errorcode "sign2c" errorlevel 1;
        sign3c: when AE = "C" and IAI = "S" then O > 0 errorcode "sign3c" errorlevel 1;
        sign4c: when AE = "C" and IAI = "IN1" then O > 0 errorcode "sign4c" errorlevel 2;
        sign9: when IAI = "D4Q" and FC = "D" and IA = "FL" then O > 0 errorcode "sign9" errorlevel 3;
        sign10: when IAI = "D45" and FC = "P" and IA = "F" then O > 0 errorcode "sign10" errorlevel 4

        end datapoint ruleset;

        DS_r := check_datapoint (sc_1, signValidation all);
        Description: 'At op check_datapoint: sc_1 has an invalid datatype expected DataSet,
                      found Scalar. Please check transformation with output dataset DS_r'

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_32'
        number_inputs = 1
        error_code = "1-4-1-9"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_33(self):
        """
        Dataset --> Dataset
        Status:
        Expression: define datapoint ruleset dpr1(variable Id_3, Me_1, Me_2) is
                        when Id_3 = "CREDIT" then Me_1 >= 0 and Me_2 >= 0 errorcode "Bad credit";
                        when Id_3 = "DEBIT" then Me_1 >= 0 and Me_2 >= 0 errorcode "Bad debit"
                    end datapoint ruleset;

                    DS_r := check_datapoint(sc_1, dpr1);
        Description: At op check_datapoint: sc_1 has an invalid datatype expected DataSet,
        found Scalar. Please check transformation with output dataset DS_r

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_33'
        number_inputs = 1
        error_code = "1-4-1-9"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_34(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := check (DS_1 >= sc_1 imbalance DS_1 - DS_2);
        Description: 

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_34'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_35(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r := check (DS_1 >= sc_2 imbalance DS_2 - sc_1);
        Description: 

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_35'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_36(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r:= DS_1 = sc_2;
        Description:

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_36'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_37(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r:= DS_1 >= sc_1;
        Description:

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_37'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_38(self):
        """
        Dataset --> Dataset
        Status:
        Expression: DS_r:= between(sc_1, 1,5);
        Description:

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_38'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_39(self):
        """
        Dataset --> Dataset
        Status:
        Expression: define operator suma (scds1 dataset, scds2 dataset)
                        returns dataset is
                            scds1 + scds2
                    end operator;

                    DS_r := suma(sc_1, sc_2);
        Description:

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_39'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_40(self):
        """
        Dataset --> Dataset
        Status:
        Expression: define operator suma (scds1 dataset, scds2 dataset)
                        returns dataset is
                            scds1 + scds2
                    end operator;

                    DS_r := suma(sc_1, sc_2);
        Description: Both scalar_datasets belongs to one input

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_40'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_41(self):
        """
        Dataset --> Dataset
        Status:
        Expression: define operator suma (ds1 dataset, scds1 dataset)
                        returns dataset is
                            ds1 + scds1
                    end operator;

                    DS_r := suma(DS_1, sc_1);
        Description:

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_41'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_42(self):
        """
        Dataset --> Dataset
        Status:
        Expression: define operator drop_identifier (ds dataset, comp component)
                        returns dataset is
                            max(ds group except comp)
                    end operator;

                    define operator suma (ds1 dataset, scds1 dataset)
                        returns dataset is
                            ds1 + scds1
                    end operator;

                    DS_r := drop_identifier (suma (DS_1, sc_1), Id_3);
        Description:

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_42'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_43(self):
        """
        Dataset --> Dataset
        Status:
        Expression: define operator drop_identifier (ds dataset, comp component)
                        returns dataset is
                            max(ds group except comp)
                    end operator;

                    define operator suma (scds1 dataset, scds2 dataset)
                        returns dataset is
                            scds1 + scds2
                    end operator;

                    DS_r := drop_identifier (suma (sc_1, sc_2), Id_3);
        Description: Component Id_3 not found. Please check transformation with output dataset DS_r

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_43'
        number_inputs = 1
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_44(self):
        """
        Dataset --> Dataset
        Status:
        Expression: define operator drop_identifier (ds dataset, comp component)
                        returns dataset is
                            max(ds group except comp)
                    end operator;

                    define operator suma (scds1 dataset, scds2 dataset)
                        returns dataset is
                            scds1 + scds2
                    end operator;

                    DS_r := drop_identifier (suma (sc_1, sc_2), sc_1);
        Description: Component sc_1 not found. Please check transformation with output dataset DS_r

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_44'
        number_inputs = 1
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_45(self):
        """
        Dataset --> Dataset
        Status:
        Expression: define operator drop_identifier (ds dataset, comp component)
                        returns dataset is
                            max(ds group except comp)
                    end operator;

                    define operator suma (ds1 dataset, scds1 dataset)
                        returns dataset is
                            ds1 + scds1
                    end operator;

                    DS_r := drop_identifier (suma (DS_1, sc_1), Id_3);
        Description:

        Git Branch: 398-scalar-tests
        Goal: 
        """
        code = 'Sc_45'
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
