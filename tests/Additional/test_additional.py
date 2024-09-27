from pathlib import Path
from typing import Union

import pytest

from tests.Helper import TestHelper
from vtlengine.API import create_ast
from vtlengine.Interpreter import InterpreterAnalyzer


class AdditionalHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"

    ds_input_prefix = "DS_"

    @classmethod
    def BaseScalarTest(cls, text: str, code: str, reference_value: Union[int, float, str]):
        '''

        '''
        if text is None:
            text = cls.LoadVTL(code)
        ast = create_ast(text)
        interpreter = InterpreterAnalyzer({})
        result = interpreter.visit(ast)
        assert result["DS_r"].value == reference_value


class StringOperatorsTest(AdditionalHelper):
    """
    Group 3
    """

    classTest = 'Additional.StringOperatorsTest'

    maxDiff = None

    def test_1(self):
        '''
        Basic behaviour for two datasets.
        '''
        text = """DS_r := DS_1 || DS_2;"""

        code = '3-1'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_2(self):
        '''
        Behaviour for two datasets with nulls.
        '''
        text = """DS_r := DS_1 || DS_2;"""

        code = '3-2'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_3(self):
        '''
        Behaviour for two datasets with diferent number of rows.
        '''
        text = """DS_r := DS_1 || DS_2;"""

        code = '3-3'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_4(self):
        '''
        Behaviour for components with nulls.
        '''
        text = """DS_r := DS_1[calc Me_3:= Me_1 || Me_2];"""

        code = '3-4'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_5(self):
        '''
        Behaviour for two datasets when the left one has more identifiers.
        '''
        text = """DS_r := DS_1 || DS_2;"""

        code = '3-5'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_6(self):
        '''
        Behaviour for two datasets when the right one has more identifiers.
        '''
        text = """DS_r := DS_1 || DS_2;"""

        code = '3-6'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_7(self):
        '''
        Behaviour for two datasets with different measures (not allowed).
        '''
        text = """DS_r := DS_1 || DS_2;"""
        code = "3-7"
        number_inputs = 2

        message = "1-1-14-1"
        self.NewSemanticExceptionTest(
            text=text,
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_11(self):
        '''
        Behaviour for dataset.
        '''
        text = """DS_r := substr(DS_1, 2, 3);"""

        code = '3-11'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_12(self):
        '''
        Behaviour for dataset.
        '''
        text = """DS_r := substr(DS_1, 2);"""

        code = '3-12'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_13(self):
        '''
        Behaviour for dataset.
        '''
        text = """DS_r := substr(DS_1, _, 3);"""

        code = '3-13'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_14(self):
        '''
        Behaviour for dataset.
        '''
        text = """DS_r := substr(DS_1);"""

        code = '3-14'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_15(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_2:= substr(Me_1)];"""

        code = '3-15'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_16(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_2:= substr(Me_1, _, 3)];"""

        code = '3-16'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_17(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_2:= substr(Me_1, 3)];"""

        code = '3-17'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_18(self):
        '''
        Behaviour for components and null.
        '''
        text = """DS_r := DS_1[calc Me_2:= substr(Me_1, null)];"""

        code = '3-18'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_19(self):
        '''
        Behaviour for components and null.
        '''
        text = """DS_r := DS_1[calc Me_2:= substr(Me_1, 1, null)];"""

        code = '3-19'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_20(self):
        '''
        Behaviour for components and null.
        '''
        text = """DS_r := DS_1[calc Me_2:= substr(Me_1, null, null)];"""

        code = '3-20'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_21(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_1:= substr(Me_1, Me_2, Me_3)];"""
        code = '3-21'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_22(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_3:= substr(Me_1, _, Me_2)];"""

        code = '3-22'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_23(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_3:= substr(Me_1, Me_2)];"""

        code = '3-23'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_26(self):
        '''
        Behaviour for dataset with null values.
        '''
        text = """DS_r := replace(DS_1, "Hello", "Hi");"""

        code = '3-26'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_27(self):
        '''
        Behaviour for dataset with null values.
        '''
        text = """DS_r := replace(DS_1, "Hello");"""

        code = '3-27'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_28(self):
        '''
        Behaviour for dataset with null.
        '''
        text = """DS_r := replace(DS_1, null, "abc");"""

        code = '3-28'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_29(self):
        '''
        Behaviour for dataset with null.
        '''
        text = """DS_r := replace(DS_1, "abc", null);"""
        code = '3-29'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_30(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_2:= replace(Me_1, "Hello")];"""
        code = '3-30'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_31(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_2:= replace(Me_1, "Hello", "Hi")];"""
        code = '3-31'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_32(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_3:= replace(Me_1, Me_2)];"""
        code = '3-32'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_33(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_4:= replace(Me_1, Me_2, Me_3)];"""
        code = '3-33'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_34(self):
        '''
        Behaviour for components with null.
        '''
        text = """DS_r := DS_1[calc Me_2:= replace(Me_1, null)];"""
        code = '3-34'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_35(self):
        '''
        Behaviour for components with null.
        '''
        text = """DS_r := DS_1[calc Me_2:= replace(Me_1, null, null)];"""
        code = '3-35'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_36(self):
        '''
        Behaviour for components with null.
        '''
        text = """DS_r := DS_1[calc Me_2:= replace(Me_1, "a", null)];"""
        code = '3-36'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_37(self):
        '''
        Behaviour for components with null.
        '''
        text = """DS_r := DS_1[calc Me_2:= replace(Me_1, null, "a")];"""
        code = '3-37'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_41(self):
        '''
        Behaviour for dataset with null values.
        '''
        text = """DS_r:= instr(DS_1, "o", 3);"""
        code = '3-41'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_42(self):
        '''
        Behaviour for dataset with null values.
        '''
        text = """DS_r:= instr(DS_1, "o", _, 2);"""
        code = '3-42'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_43(self):
        '''
        Behaviour for dataset with null values.
        '''
        text = """DS_r:= instr(DS_1, "o", 4, 3);"""
        code = '3-43'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_44(self):
        '''
        Behaviour for dataset with null.
        '''
        text = """DS_r:= instr(DS_1, null, 4, 3);"""
        code = '3-44'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_45(self):
        '''
        Behaviour for dataset with null.
        '''
        text = """DS_r:= instr(DS_1, "s", null, 3);"""
        code = '3-45'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_46(self):
        '''
        Behaviour for dataset with null.
        '''
        text = """DS_r:= instr(DS_1, "s", 3, null);"""
        code = '3-46'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_47(self):
        '''
        Behaviour for component with null values.
        '''
        text = """DS_r := DS_1[calc Me_2:=instr(Me_1,"o", 3)];"""
        code = '3-47'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_48(self):
        '''
        Behaviour for component with null values.
        '''
        text = """DS_r := DS_1[calc Me_2:=instr(Me_1,"o", _, 2)];"""
        code = '3-48'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_49(self):
        '''
        Behaviour for component with null values.
        '''
        text = """DS_r := DS_1[calc Me_2:=instr(Me_1,"o", 6, 4)];"""
        code = '3-49'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_50(self):
        '''
        Behaviour for component with null.
        '''
        text = """DS_r := DS_1[calc Me_2:=instr(Me_1, null, 6, 4)];"""
        code = '3-50'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_51(self):
        '''
        Behaviour for component with null.
        '''
        text = """DS_r := DS_1[calc Me_2:=instr(Me_1, "", null, 4)];"""

        code = '3-51'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_52(self):
        '''
        Behaviour for component with null.
        '''
        text = """DS_r := DS_1[calc Me_2:=instr(Me_1, "o", 4, null)];"""
        code = '3-52'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_53(self):
        '''
        Behaviour for components with null values.
        '''
        text = """DS_r := DS_1[calc Me_3:=instr(Me_1, Me_2)];"""
        code = '3-53'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_54(self):
        '''
        Behaviour for components with null values.
        '''
        text = """DS_r := DS_1[calc Me_4:=instr(Me_1, Me_2, Me_3)];"""
        code = '3-54'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_55(self):
        '''
        Behaviour for components with null values.
        '''
        text = """DS_r := DS_1[calc Me_4:=instr(Me_1, Me_2, _, Me_3)];"""
        code = '3-55'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_56(self):
        '''
        Behaviour for components with null values.
        '''
        text = """DS_r := DS_1[calc Me_5:=instr(Me_1, Me_2, Me_3, Me_4)];"""
        code = '3-56'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)


class NumericOperatorsTest(AdditionalHelper):
    """
    Group 4
    """

    classTest = 'Additional.NumericOperatorsTest'

    maxDiff = None

    def test_4(self):
        '''
        Null Unary operations ('+') with Datasets.
        '''
        text = """DS_r := +DS_1;"""

        code = '4-4'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_9(self):
        '''
        Basic behaviour for dataset.
        '''
        text = """DS_r := round(DS_1);"""

        code = '4-9'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_10(self):
        '''
        Basic behaviour for dataset with null values.
        '''
        text = """DS_r := round(DS_1, 0);"""

        code = '4-10'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_11(self):
        '''
        Basic behaviour for dataset with null.
        '''
        text = """DS_r := round(DS_1, null);"""

        code = '4-11'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_12(self):
        '''
        Basic behaviour for components with null values.
        '''
        text = """DS_r := DS_1 [ calc Me_3:= round(Me_1, Me_2) ];"""

        code = '4-12'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_15(self):
        '''
        Basic behaviour for components with null values.
        '''
        text = """DS_r := trunc(DS_1, 0);"""

        code = '4-15'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_16(self):
        '''
        Basic behaviour for components with null values.
        '''
        text = """DS_r := trunc(DS_1, null);"""

        code = '4-16'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_17(self):
        '''
        Basic behaviour for components with null values.
        '''
        text = """DS_r := DS_1 [ calc Me_3:= trunc(Me_1, Me_2) ];"""

        code = '4-17'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_20(self):
        '''
        Basic behaviour for components.
        '''
        text = """DS_r  := DS_1[ calc Me_3 := power(Me_1, Me_2) ];"""

        code = '4-20'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_21(self):
        '''
        Basic behaviour for dataset and null.
        '''
        text = """DS_r  := power(DS_1, null);"""

        code = '4-21'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_22(self):
        '''
        Basic behaviour for dataset with null values.
        '''
        text = """DS_r  := power(DS_1, 2);"""

        code = '4-22'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_23(self):
        '''
        Basic behaviour for component and null.
        '''
        text = """DS_r  := DS_1[ calc Me_2 := power(Me_1, null) ];"""

        code = '4-23'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_24(self):
        '''
        Basic behaviour for components with null values.
        '''
        text = """DS_r  := DS_1[ calc Me_2 := power(Me_1, 2) ];"""

        code = '4-24'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_27(self):
        '''
        Basic behaviour for components.
        '''
        text = """DS_r  := DS_1[ calc Me_3 := log(Me_1, Me_2) ];"""

        code = '4-27'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_28(self):
        '''
        Basic behaviour for dataset and null.
        '''
        text = """DS_r  := log(DS_1, null);"""

        code = '4-28'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_29(self):
        '''
        Basic behaviour for dataset with null values.
        '''
        text = """DS_r  := log(DS_1, 2);"""

        code = '4-29'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_30(self):
        '''
        Basic behaviour for component and null.
        '''
        text = """DS_r  := DS_1[ calc Me_2 := log(Me_1, null) ];"""

        code = '4-30'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_31(self):
        '''
        Basic behaviour for components with null values.
        '''
        text = """DS_r  := DS_1[ calc Me_2 := log(Me_1, 2) ];"""

        code = '4-31'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)


class ComparisonOperatorsTest(AdditionalHelper):
    """
    Group 5
    """

    classTest = 'Additional.ComparisonOperatorsTest'

    maxDiff = None

    def test_2(self):
        '''

        '''
        text = """DS_r := exists_in (DS_1, DS_2);"""
        code = '5-2'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_3(self):
        '''
        equal to reference manual test but this at DS_1 contains nulls.
        '''
        text = """DS_r := exists_in (DS_1, DS_2, all);"""
        code = '5-3'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_4(self):
        '''
        equal to reference manual test but this at DS_1 contains nulls.
        '''
        text = """DS_r := exists_in (DS_1, DS_2, true);"""
        code = '5-4'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_5(self):
        '''
        equal to reference manual test but this at DS_1 contains nulls.
        '''
        text = """DS_r := exists_in (DS_1, DS_2, false);"""
        code = '5-5'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_6(self):
        '''
        equal to test 2 but this at DS_1 contains nulls.
        '''
        text = """DS_r := exists_in (DS_1, DS_2);"""
        code = '5-6'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_7(self):
        '''
        equal to reference manual test but this with reverse order.
        '''
        text = """DS_r := exists_in (DS_2, DS_1, all);"""
        code = '5-7'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_9(self):
        '''
        equal to reference manual test but at this one DS_2 have no Id_4 (different number of Ids).
        '''
        text = """DS_r := exists_in (DS_2, DS_1, all);"""
        code = '5-9'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_10(self):
        '''
        Behaviour for dataset with null values and scalars.
        '''
        text = """DS_r:= between(DS_1, 5, 10);"""
        code = '5-10'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_11(self):
        '''
        Behaviour for dataset with null scalars.
        '''
        text = """DS_r:= between(DS_1, 5, null);"""
        code = '5-11'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_12(self):
        '''
        Behaviour for dataset with null scalars.
        '''
        text = """DS_r:= between(DS_1, null, 10);"""
        code = '5-12'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_13(self):
        '''
        Behaviour for dataset with null scalars.
        '''
        text = """DS_r:= between(DS_1, null, null);"""
        code = '5-13'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_16(self):
        '''
        Behaviour for components with null values.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, Me_2, Me_3) ];"""
        code = '5-16'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_17(self):
        '''
        Behaviour for components with null values and scalar.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, Me_2, 100) ];"""
        code = '5-17'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_18(self):
        '''
        Behaviour for components with null values and null scalar.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, Me_2, null) ];"""
        code = '5-18'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_19(self):
        '''
        Behaviour for components with null values and null scalar.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, null, Me_2) ];"""
        code = '5-19'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_20(self):
        '''
        Behaviour for components with null values and scalar.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, 900, Me_2) ];"""
        code = '5-20'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_21(self):
        '''
        Behaviour for component with null values and scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, 5, 300) ];"""
        code = '5-21'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_22(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, null, null) ];"""
        code = '5-22'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_23(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, 4, null) ];"""
        code = '5-23'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_24(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, null, 4) ];"""
        code = '5-24'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_25(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(null, Me_1, 4) ];"""
        code = '5-25'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_26(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(null, Me_1, null) ];"""
        code = '5-26'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_27(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(4, Me_1, null) ];"""
        code = '5-27'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_28(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(null, 4, Me_1) ];"""
        code = '5-28'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_29(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(null, null, Me_1) ];"""
        code = '5-29'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_30(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(4, null, Me_1) ];"""
        code = '5-30'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_31(self):
        '''
        Behaviour for component with null values and scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(6, 1, Me_1) ];"""
        code = '5-31'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_32(self):
        '''
        Behaviour for component with null values and scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(6, Me_1, 10) ];"""
        code = '5-32'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_33(self):
        '''
        Behaviour for components with null values and scalar.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(6, Me_1, Me_2) ];"""
        code = '5-33'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_34(self):
        '''
        Behaviour for components with null values and null scalar.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(null, Me_1, Me_2) ];"""
        code = '5-34'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)


class BooleanOperatorsTest(AdditionalHelper):
    """
    Group 6
    """

    classTest = 'Additional.BooleanOperatorsTest'

    maxDiff = None

    pass


class ClauseOperatorsTest(AdditionalHelper):
    """
    Group 13
    """

    classTest = 'Additional.ClauseOperatorsTest'

    def test_1(self):
        '''
        Basic behaviour for dataset with null values.
        '''
        text = """DS_r := DS_1 [ unpivot Id_2, Me_1];"""

        code = '13-1'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_2(self):
        '''
        Behaviour for unpivot mixed with another operator.
        '''
        text = """DS_r := DS_1 [ unpivot Id_2, Me_1] + DS_2;"""

        code = '13-2'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_3(self):
        '''
        Create null measures casting the value.
        '''
        text = """DS_r := DS_1[calc Me_10 := cast(null, number)];"""

        code = '13-3'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_4(self):
        '''
        Replace null measures casting the value.
        '''
        # Load the files
        text = """DS_r := DS_1[calc Me_1 := cast(null, string)];"""

        code = '13-4'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_6(self):
        '''

        '''
        text = """DS_r := DS_1 [ aggr Me_4:= sum( Me_1 ), Me_2 := max( Me_1) group by Id_1 , Id_2 ][calc Me_6:= 2];"""

        code = '13-6'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    # unpivot
    def test_GL_49_1(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [unpivot Id_2, Me_3];
        Description: Unpivot that result has nulls
            line RM 7200: " When a Measure is NULL then unpivot does not create
                            a Data Point for that Measure."
        Git Branch: bug-49-unpivot-and-nulls.
        Goal: Check Result.
        """
        code = 'GL_49_1'
        number_inputs = 1
        references_names = ["DS_r"]
        text = "DS_r := DS_1 [unpivot Id_2, Me_3];"

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            text=text
        )

    # OK
    def test_GL_49_2(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [unpivot Id_2, Me_3];
        Description: two measures, one measure is all null Unpivot that result has nulls
            line RM 7200: " When a Measure is NULL then unpivot does not create
                            a Data Point for that Measure."
        Git Branch: bug-49-unpivot-and-nulls.
        Goal: Check Result.
        """
        code = 'GL_49_2'
        number_inputs = 1
        references_names = ["DS_r"]
        text = "DS_r := DS_1 [unpivot Id_2, Me_3];"

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            text=text
        )

    # OK
    def test_GL_49_3(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [unpivot Id_2, Me_3];
        Description: only one measure, is all null Unpivot that result has nulls
            line RM 7200: " When a Measure is NULL then unpivot does not create
                            a Data Point for that Measure."
        Git Branch: bug-49-unpivot-and-nulls.
        Goal: Check Result.
        """
        code = 'GL_49_3'
        number_inputs = 1
        references_names = ["DS_r"]
        text = "DS_r := DS_1 [unpivot Id_2, Me_3];"

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            text=text
        )

    def test_GL_49_4(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [unpivot Id_3, Me_3];
        Description: with several identifiers as input.
            line RM 7200: " When a Measure is NULL then unpivot does not create
                            a Data Point for that Measure."
        Git Branch: bug-49-unpivot-and-nulls.
        Goal: Check Result.
        """
        code = 'GL_49_4'
        number_inputs = 1
        references_names = ["DS_r"]
        text = "DS_r := DS_1 [unpivot Id_3, Me_3];"

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            text=text
        )

    def test_GL_49_6(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [unpivot Id_3, Me_3];
        Description: unpivot with measure with same name.
            line RM 7200: " When a Measure is NULL then unpivot does not create
                            a Data Point for that Measure."
        Git Branch: bug-49-unpivot-and-nulls.
        Goal: Check Result.
        """
        code = 'GL_49_6'
        number_inputs = 1
        references_names = ["DS_r"]
        text = "DS_r := DS_1 [unpivot Id_3, Me_2];"

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            text=text
        )


class SetOperatorsTest(AdditionalHelper):
    """
    Group 8
    """

    classTest = 'Additional.SetOperatorsTest'

    maxDiff = None

    def test_1(self):
        '''
        Basic behaviour.
        '''
        text = """DS_r := intersect(DS_1,DS_2);"""

        code = '8-1'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_2(self):
        '''
        Basic behaviour.
        '''
        text = """DS_r := setdiff(DS_1,DS_2);"""

        code = '8-2'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_3(self):
        '''
        Basic behaviour.
        '''
        text = """DS_r := symdiff(DS_1,DS_2);"""

        code = '8-3'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_4(self):
        '''
        Basic behaviour.
        '''
        text = """DS_r := intersect(DS_1,DS_2,DS_3);"""

        code = '8-4'
        number_inputs = 3
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_5(self):
        '''
        Basic behaviour.
        '''
        text = """DS_r := intersect(DS_1[drop Me_3] ,DS_2);"""

        code = '8-5'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_6(self):
        '''
        Basic behaviour.
        '''
        text = """DS_r := intersect(DS_1 ,DS_2);"""

        code = '8-6'
        number_inputs = 2
        message = "1-1-17-1"
        self.NewSemanticExceptionTest(
            text=text,
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_7(self):
        '''
        Basic behaviour.
        '''
        text = """DS_r := intersect(DS_1 [drop Me_1] ,DS_2);"""

        code = '8-7'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_8(self):
        '''
        Basic behaviour.
        '''
        text = """DS_r := intersect(DS_1 ,DS_2);"""

        code = '8-8'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_9(self):
        '''
        Basic behaviour.
        Description: Empty result.
        '''
        text = """DS_r := intersect(DS_1 ,DS_2);"""

        code = '8-9'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)


class AggregateOperatorsTest(AdditionalHelper):
    """
    Group 10
    """

    classTest = 'Additional.AggregateOperatorsTest'

    maxDiff = None

    def test_1(self):
        '''
        Basic behaviour for datasets.
        '''
        text = """DS_r := count(DS_1 group by Id_1);"""
        code = '10-1'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_2(self):
        '''
        Basic behaviour for datasets.
        '''
        text = """DS_r := count(DS_1 group by Id_1, Id_2);"""
        code = '10-2'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_3(self):
        '''
        feat-VTLEN-532-No-measures-agg
        feat: no measures agg for count,min and max
        '''
        text = """DS_r := min(DS_1 group by DT_RFRNC, PRSPCTV_ID);"""
        code = '10-3'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_4(self):
        '''
        feat-VTLEN-532-No-measures-agg
        feat: no measures agg for count,min and max
        '''
        text = """DS_r := max(DS_1 group by DT_RFRNC, PRSPCTV_ID);"""
        code = '10-4'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_5(self):
        '''
        feat-VTLEN-532-No-measures-agg
        feat: no measures agg for count,min and max
        '''
        text = """DS_r := count(DS_1 group by DT_RFRNC, PRSPCTV_ID);"""
        code = '10-5'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_6(self):
        '''
        feat-VTLEN-532-No-measures-agg
        feat: no measures agg for count,min and max
        '''
        text = """DS_r := median(DS_1 group by DT_RFRNC, PRSPCTV_ID);"""
        code = '10-6'
        number_inputs = 1
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(
            text=text,
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_7(self):
        '''
        Description: Recheck of the no measures agg
        Jira issue: bug VTLEN 536.
        Git Branch: bug-VTLEN-546-Max-no-measures.
        Goal: Interpreter results.
        '''
        text = """BNFCRS_TRNSFRS_CMMN_INSTRMNTS_3 :=
                    max(BNFCRS_TRNSFRS_CMMN_INSTRMNTS_2 
                    group by BNFCRS_CNTRPRTY_ID,
                            TRNSFR_CNTRPRTY_ID,
                            BNFCRS_DT_RFRNC,
                            BNFCRS_INSTRMNT_UNQ_ID,
                            BNFCRS_PRSPCTV_ID);

                BNFCRS_TRNSFRS_CMMN_INSTRMNTS_4 :=
                    BNFCRS_TRNSFRS_CMMN_INSTRMNTS_3
                        [rename BNFCRS_DT_RFRNC to DT_RFRNC,
                                BNFCRS_INSTRMNT_UNQ_ID to INSTRMNT_UNQ_ID,
                                BNFCRS_PRSPCTV_ID to PRSPCTV_ID]
                        [calc BNFCR_ID := BNFCRS_CNTRPRTY_ID,
                            TRNSFR_ID := TRNSFR_CNTRPRTY_ID];"""
        code = '10-7'
        number_inputs = 2
        references_names = ["DS_r1", "DS_r2"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_GL_222_1(self):
        '''
        '''
        text = """DS_r := DS_1[aggr Me_3 := count ( ) , Me_4 := count ( ) group by Id_1];"""
        code = 'GL_222_1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)


class JoinOperatorsTest(AdditionalHelper):
    """
    Group 2
    """

    classTest = 'Additional.JoinOperatorsTest'

    def test_1(self):
        '''

        '''
        text = """DS_r := inner_join ( DS_1, DS_2 keep Me_1);"""
        code = '2-1'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_2(self):
        '''

        '''
        text = """DS_r := inner_join ( DS_1 as ds1, DS_2 as ds2 keep Me_1);"""
        code = '2-2'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_3(self):
        '''

        '''
        text = """DS_r := inner_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_1#Me_2 keep Me_1, DS_1#Me_2);"""
        code = '2-3'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_4(self):
        '''

        '''
        text = """DS_r := inner_join ( DS_1 as d1, DS_2 as d2 filter Me_1 ="A" keep Me_1, d1#Me_2 );"""
        code = '2-4'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_5(self):
        '''

        '''
        text = """DS_r := left_join ( DS_1 as d1, DS_2 as d2  rename d2#Me_2 to ent1, d1#Me_2 to ent2);"""
        code = '2-5'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_6(self):
        '''

        '''
        text = """DS_r := inner_join ( DS_1 as d1, DS_2[keep Me_1A, Me_2] as d2 keep d2#Me_2);"""
        code = '2-6'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_7(self):
        '''

        '''
        text = """DS_r := inner_join ( DS_1, DS_2[keep Me_1A, Me_2] as d2 keep DS_1#Me_2);"""
        code = '2-7'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_8(self):
        '''

        '''
        text = """DS_r := inner_join ( DS_1  filter Id_2="B" calc Me_3:=Me_1);"""
        code = '2-8'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_9(self):
        '''

        '''
        text = """DS_r := left_join ( DS_1, DS_2 using  Id_1, Me_2);"""
        code = '2-9'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_10(self):
        '''

        '''
        text = """DS_r := inner_join ( DS_1, DS_2 using  Id_1, Me_2);"""
        code = '2-10'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_11(self):
        '''

        '''
        text = """DS_r := inner_join ( DS_1, DS_2 rename DS_1#Me_2 to Me_4);"""
        code = '2-11'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_12(self):
        '''

        '''
        text = """DS_r := inner_join ( DS_1, DS_2 rename Me_2 to Me_4);"""
        code = '2-12'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_13(self):
        '''
        reverse order of test 10.
        '''
        text = """DS_r := inner_join ( DS_2, DS_1 using  Id_1, Me_2);"""
        code = '2-10'
        number_inputs = 2
        references_names = ["DS_r"]

        # Bad using clause, it not defines ids from DS_1
        # self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)
        assert True

    def test_15(self):
        '''
        Only identifiers inner_join Interpreter.
        '''
        text = """DS_r <- inner_join(DS_1, DS_2);"""
        code = '2-15'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_16(self):
        '''
        Description: Left_join review.
        Jira issue: VTLEN 540.
        Git Branch: feat-VTLEN-540-left_join-review.
        Goal: Check Semantic Exception.
        '''
        text = """DS_r := left_join(A, B using Id1, IdB2, IdB3);"""
        code = '2-16'
        number_inputs = 2
        error_code = "1-1-13-6"
        self.NewSemanticExceptionTest(
            text=text,
            code=code,
            number_inputs=number_inputs,
            exception_code=error_code
        )

    def test_19(self):
        '''
        INNER JOIN OPERATOR
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 );
        Description: Inner_join supersets review.
        Jira issue: VTLEN 588.
        Git Branch: feat-VTLEN-588-Aditional-join-tests.
        Goal: Check Result.
        '''
        text = """DS_r := inner_join ( DS_1, DS_2 );"""
        code = '2-19'
        number_inputs = 2
        references_names = ["DS_r"]

        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_20(self):
        '''
        Status: OK
        Description: Inner_join supersets review.
        Jira issue: VTLEN 588.
        Git Branch: feat-VTLEN-588-Aditional-join-tests.
        Goal: Check Result.
        '''
        text = """DS_r := inner_join ( DS_3, DS_1, DS_2 );"""
        code = '2-20'
        number_inputs = 3
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_22(self):
        '''
        Status: OK
        Description: Full_join supersets review.
        Jira issue: VTLEN 588.
        Git Branch: feat-VTLEN-588-Aditional-join-tests.
        Goal: Check Result.
        '''
        text = """DS_r := full_join ( DS_1, DS_2 );"""
        code = '2-22'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_23(self):
        '''
        Status: OK
        Description: Full_join supersets review.
        Jira issue: VTLEN 588.
        Git Branch: feat-VTLEN-588-Aditional-join-tests.
        Goal: Check Exception.
        '''
        text = """DS_r := full_join ( DS_1, DS_2 );"""
        code = '2-23'
        number_inputs = 2
        message = "1-1-13-13"
        self.NewSemanticExceptionTest(
            text=text,
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_25(self):
        '''
        Status: OK
        Description: Inner_join using review.
        Jira issue: VTLEN 588.
        Git Branch: feat-VTLEN-588-Aditional-join-tests.
        Goal: Check Exception, should be an exception.
        Further description: Ver conjuntamente con el 26 pues los dos son tipo B1 (2263) por 2444-2455 debe dar error pero me falla el rename tb(2308-2316) y no debería
        Tb se ve el desdoblamiento de Id_2.
        '''
        text = """DS_r := inner_join ( DS_1, DS_2 using Id_1 );"""
        code = '2-25'
        number_inputs = 2
        message = "1-1-13-4"
        self.NewSemanticExceptionTest(
            text=text,
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_26(self):
        '''
        Status: OK
        Description: Inner_join using review.
        Jira issue: VTLEN 588.
        Git Branch: feat-VTLEN-588-Aditional-join-tests.
        Goal: Check Exception, should be an exception.
        Further description: At op inner_join: Using clause does not define all the identifiers of non reference datasets.
        '''
        text = """DS_r := inner_join ( DS_1, DS_2 using Id_2 );"""
        code = '2-26'
        number_inputs = 2
        message = "1-1-13-4"
        self.NewSemanticExceptionTest(
            text=text,
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_27(self):
        '''
        Status: OK
        Description: Inner_join using review.
        Jira issue: VTLEN 588.
        Git Branch: feat-VTLEN-588-Aditional-join-tests.
        Goal: Check Result.
        Further description: 3 Datasets.
        '''
        text = """DS_r := inner_join ( DS_1, DS_2, DS_3 using Id_1 );"""
        code = '2-27'
        number_inputs = 3
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_28(self):
        '''
        Status: OK
        Description: Inner_join using review.
        Jira issue: VTLEN 588.
        Git Branch: feat-VTLEN-588-Aditional-join-tests.
        Goal: Check Exception.
        Further description: 3 Datasets.
        '''
        text = """DS_r := inner_join ( DS_1, DS_2, DS_3 using Id_1 );"""
        code = '2-28'
        number_inputs = 3
        message = "1-1-13-4"
        self.NewSemanticExceptionTest(
            text=text,
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_29(self):
        '''
        Status: OK
        Description: Inner_join using review.
        Jira issue: VTLEN 588.
        Git Branch: feat-VTLEN-588-Aditional-join-tests.
        Goal: Check Exception.
        Further description: The same that 28 but with two datasets that can act as reference.
        '''
        text = """DS_r := inner_join ( DS_1, DS_2, DS_3 using Id_1 );"""
        code = '2-29'
        number_inputs = 3
        message = "1-1-13-4"
        self.NewSemanticExceptionTest(
            text=text,
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_30(self):
        '''
        Status: OK
        Description: Left_join using review.
        Jira issue: VTLEN 588.
        Git Branch: feat-VTLEN-588-Aditional-join-tests.
        Goal: Check Exception.
        '''
        text = """DS_r := left_join ( DS_1, DS_2 using Id_1 );"""
        code = '2-30'
        number_inputs = 2
        message = "1-1-13-4"
        self.NewSemanticExceptionTest(
            text=text,
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_31(self):
        '''
        Status: OK
        Description: Left_join using review.
        Jira issue: VTLEN 588.
        Git Branch: feat-VTLEN-588-Aditional-join-tests.
        Goal: Check Result.
        Further description: DS1 contains all the DS2's ids.
        '''
        text = """DS_r := left_join ( DS_1, DS_2 using Id_1, Id_2 );"""
        code = '2-31'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_32(self):
        '''
        Status: OK
        Description: left_join using review.
        Jira issue: VTLEN 588.
        Git Branch: feat-VTLEN-588-Aditional-join-tests.
        Goal: Check Result.
        Further description: 31 with different order.
        '''
        text = """DS_r := left_join ( DS_2, DS_1 using Id_1, Id_2 );"""
        code = '2-32'
        number_inputs = 2
        message = "1-1-13-4"
        self.NewSemanticExceptionTest(
            text=text,
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_33(self):
        '''
        Status: OK
        Description: Left_join using review.
        Jira issue: VTLEN 588.
        Git Branch: feat-VTLEN-588-Aditional-join-tests.
        Goal: Check Result.
        Further description: 3 Datasets.
        '''
        text = """DS_r := left_join ( DS_1, DS_2, DS_3 using Id_1 );"""
        code = '2-33'
        number_inputs = 3
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_34(self):
        '''
        Status: OK
        Description: left_join using review.
        Jira issue: VTLEN 588.
        Git Branch: feat-VTLEN-588-Aditional-join-tests.
        Goal: Check Result.
        Further description: 3 Datasets.
        '''
        text = """DS_r := left_join ( DS_1, DS_2, DS_3 using Id_1 );"""
        code = '2-34'
        number_inputs = 3
        message = "1-1-13-4"
        self.NewSemanticExceptionTest(
            text=text,
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_35(self):
        '''
        Status: DOUBT
        Description: left_join using review.
        Jira issue: VTLEN 588.
        Git Branch: feat-VTLEN-588-Aditional-join-tests.
        Goal: Check Result.
        Further description: Should be a b1? i have doubts if this is possible.
        -b1 is a kind of join, read the manual-
        '''
        text = """DS_r := left_join ( DS_1, DS_2 using Id_1 );"""
        code = '2-35'
        number_inputs = 2
        message = "1-1-13-4"
        self.NewSemanticExceptionTest(
            text=text,
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    # TODO: Move to a better place

    def test_36(self):
        '''
        Status: OK
        Description: Left_join using review.
        Git issue: 42-create-null-implicit-promotion-at-left-join-op.
        Git Branch: feat-42-create-null-implicit-promotion-at-left-join-op.
        Goal: Check Result.
        Further description: 3 Datasets.
        '''
        text = """DS_r := left_join(DS_1, DS_2, DS_3 rename DS_2#Me_1 to Me_1_2, DS_2#Me_2 to Me_2_2, DS_3#Me_1 to Me_1_3, DS_3#Me_2 to Me_2_3);"""
        code = '2-36'
        number_inputs = 3
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_37(self):
        '''
        Status: OK
        Description: Left_join using review.
        Git issue: 42-create-null-implicit-promotion-at-left-join-op.
        Git Branch: feat-42-create-null-implicit-promotion-at-left-join-op.
        Goal: Check Result.
        Further description: 3 Datasets.
        '''
        text = """DS_r := left_join(DS_1, DS_2, DS_3 using Id_2, Id_1);"""
        code = '2-37'
        number_inputs = 3
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_38(self):
        '''
        Status: OK
        Description: full_join implicit promotion review.
        Git issue: 42-create-null-implicit-promotion-at-left-join-op.
        Git Branch: feat-42-create-null-implicit-promotion-at-left-join-op.
        Goal: Check Result.
        Further description: 2 Datasets.
        '''
        text = """DS_r := full_join(DS_1, DS_2 keep DS_2#Me_1, Me_3, Me_2);"""
        code = '2-38'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    # BUG
    def test_39(self):
        '''
        Status: Differences between Interpreter and Semantic, for the rename
        Description: full_join implicit promotion review.
        Git issue: 42-create-null-implicit-promotion-at-left-join-op.
        Git Branch: feat-42-create-null-implicit-promotion-at-left-join-op.
        Goal: Check Result.
        Further description: 3 Datasets.
        '''
        text = """DS_r := full_join(DS_1 as d1, DS_2 [rename Me_1 to Me_1_2, Me_2 to Me_2_2] as d2, DS_3 [rename Me_1 to Me_1_3, Me_2 to Me_2_3] as d3);"""
        code = '2-39'
        number_inputs = 3
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_40(self):
        '''
        Status: OK
        Description: full_join implicit promotion review.
        Git issue: 42-create-null-implicit-promotion-at-left-join-op.
        Git Branch: feat-42-create-null-implicit-promotion-at-left-join-op.
        Goal: Check Result.
        Further description: 3 Datasets.
        '''
        text = """DS_r := full_join(DS_1 as d1, DS_2 as d2, DS_3 as d3);"""
        code = '2-40'
        number_inputs = 3
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)


class DataValidationOperatorsTest(AdditionalHelper):
    """
    Group 11
    """

    classTest = 'Additional.DataValidationOperatorsTest'

    maxDiff = None

    def test_1(self):
        '''

        '''
        text = """DS_r := check ( DS_1 >= DS_2);"""

        code = '11-1'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_2(self):
        '''

        '''
        text = """DS_r := check ( DS_1 >= DS_2 errorcode 1.0 errorlevel "local" imbalance DS_1 - DS_2 );"""

        code = '11-2'
        number_inputs = 2
        references_names = ["DS_r"]

        with pytest.raises(Exception, match="Error level must be an integer, line 1"):
            self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                          references_names=references_names)

    def test_3(self):
        '''

        '''
        text = """DS_r := check ( DS_1 >= DS_2 imbalance DS_1 - DS_2 )#bool_var = (DS_1 >= DS_2);"""

        code = '11-3'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_4(self):
        '''
        DAG Error: R070, R020 and R110 generate a cycle.
        '''
        text = """define hierarchical ruleset HR_1 ( variable rule Id_2 ) is
                R010 : A = J + K + L                        errorlevel 5 ;
                R020 : B = M + N + O                        errorlevel 5 ;
                R030 : C = P + Q        errorcode "XX"      errorlevel 5 ;
                R040 : D = R + S                            errorlevel 1 ;
                R060 : F = Y + W + Z                        errorlevel 7 ;
                R070 : G = B + C                                         ;
                R080 : H = D + E                            errorlevel 0 ;
                R090 : I = D + G        errorcode "YY"      errorlevel 0 ;
                R100 : M >= N                               errorlevel 5 ;
                R110 : M <= G                               errorlevel 5
            end hierarchical ruleset;

            DS_r := check_hierarchy ( DS_1, HR_1 rule Id_2 all);"""

        code = '11-4'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_5(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 all);"""

        code = '11-5'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_6(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 non_zero all);"""

        code = '11-6'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_7(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 partial_null all);"""

        code = '11-7'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_8(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 partial_zero all);"""

        code = '11-8'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_9(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 always_null all);"""

        code = '11-9'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_10(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 always_zero all);"""

        code = '11-10'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_11(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 all);"""

        code = '11-11'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_12(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 non_zero all);"""

        code = '11-12'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_13(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 partial_null all);"""

        code = '11-13'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_14(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 partial_zero all);"""

        code = '11-14'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_15(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 always_null all);"""

        code = '11-15'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_16(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 always_zero all);"""

        code = '11-16'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_17(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 all);"""

        code = '11-17'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_18(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 non_zero all);"""

        code = '11-18'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_19(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 partial_null all);"""

        code = '11-19'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_20(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 partial_zero all);"""

        code = '11-20'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_21(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 always_null all);"""

        code = '11-21'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_22(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 always_zero all);"""

        code = '11-22'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_23(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 partial_null all);"""

        code = '11-23'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_24(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A=B+C   errorcode "error"   errorlevel 5;
                    A>=B    errorcode "error2"  errorlevel 5;
                    A>=C    errorcode "error3"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 partial_zero all);"""

        code = '11-24'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_25(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A = B + C    errorcode "error"   errorlevel 5;
                    A >= B       errorcode "error2"  errorlevel 5;
                    A >= C       errorcode "error3"  errorlevel 5;
                    A = E + F    errorcode "error4"  errorlevel 5;
                    D = E + F    errorcode "error5"  errorlevel 5;
                    C = E + F    errorcode "error6"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 all);"""

        code = '11-25'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_26(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A = B + C    errorcode "error"   errorlevel 5;
                    A >= B       errorcode "error2"  errorlevel 5;
                    A >= C       errorcode "error3"  errorlevel 5;
                    A = E + F    errorcode "error4"  errorlevel 5;
                    D = E + F    errorcode "error5"  errorlevel 5;
                    C = E + F    errorcode "error6"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 non_zero all);"""

        code = '11-26'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_27(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A = B + C    errorcode "error"   errorlevel 5;
                    A >= B       errorcode "error2"  errorlevel 5;
                    A >= C       errorcode "error3"  errorlevel 5;
                    A = E + F    errorcode "error4"  errorlevel 5;
                    D = E + F    errorcode "error5"  errorlevel 5;
                    C = E + F    errorcode "error6"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 partial_null all);"""

        code = '11-27'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_28(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A = B + C    errorcode "error"   errorlevel 5;
                    A >= B       errorcode "error2"  errorlevel 5;
                    A >= C       errorcode "error3"  errorlevel 5;
                    A = E + F    errorcode "error4"  errorlevel 5;
                    D = E + F    errorcode "error5"  errorlevel 5;
                    C = E + F    errorcode "error6"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 partial_zero all);"""

        code = '11-28'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_29(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A = B + C    errorcode "error"   errorlevel 5;
                    A >= B       errorcode "error2"  errorlevel 5;
                    A >= C       errorcode "error3"  errorlevel 5;
                    A = E + F    errorcode "error4"  errorlevel 5;
                    D = E + F    errorcode "error5"  errorlevel 5;
                    C = E + F    errorcode "error6"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 always_null all);"""

        code = '11-29'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_30(self):
        '''

        '''
        text = """define hierarchical ruleset hie1 (variable rule Id2) is
                    A = B + C    errorcode "error"   errorlevel 5;
                    A >= B       errorcode "error2"  errorlevel 5;
                    A >= C       errorcode "error3"  errorlevel 5;
                    A = E + F    errorcode "error4"  errorlevel 5;
                    D = E + F    errorcode "error5"  errorlevel 5;
                    C = E + F    errorcode "error6"  errorlevel 5
                end hierarchical ruleset;

                DS_r := check_hierarchy(DS_1, hie1 rule Id2 always_zero all);"""

        code = '11-30'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)


class TimeOperatorsTest(AdditionalHelper):
    """
    Group 7
    """

    classTest = 'Additional.TimeOperatorsTest'

    maxDiff = None

    def test_1(self):
        '''
        Basic behaviour for datasets.
        '''
        text = """DS_r := period_indicator(DS_1);"""
        code = '7-1'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_2(self):
        '''
        Basic behaviour for datasets with date type.
        '''
        text = """DS_r := fill_time_series(DS_1, single);"""
        code = '7-2'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_3(self):
        '''
        Basic behaviour for datasets with date type.
        '''
        text = """DS_r := fill_time_series(DS_1, all);"""
        code = '7-3'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_4(self):
        '''
        Basic behaviour for datasets with date type.
        '''
        text = """DS_r := fill_time_series(DS_1);"""
        code = '7-4'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_5(self):
        '''
        Basic behaviour for datasets with period type.
        '''
        text = """DS_r := fill_time_series(DS_1, single);"""
        code = '7-5'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_6(self):
        '''
        Basic behaviour for datasets with period type.
        '''
        text = """DS_r := fill_time_series(DS_1, all);"""
        code = '7-6'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_7(self):
        '''
        Basic behaviour for datasets with period type.
        '''
        text = """DS_r := fill_time_series(DS_1);"""
        code = '7-7'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_8(self):
        '''
        Basic behaviour for datasets with date type.
        '''
        text = """DS_r := flow_to_stock(DS_1);"""
        code = '7-8'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_9(self):
        '''
        Basic behaviour for datasets with period type.
        '''
        text = """DS_r := flow_to_stock(DS_1);"""
        code = '7-9'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_10(self):
        '''
        Basic behaviour for datasets with date type.
        '''
        text = """DS_r := stock_to_flow(DS_1);"""
        code = '7-10'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_11(self):
        '''
        Basic behaviour for datasets with period type.
        '''
        text = """DS_r := stock_to_flow(DS_1);"""
        code = '7-11'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_12(self):
        '''
        Basic behaviour for datasets with date type.
        '''
        text = """DS_r := timeshift(DS_1, 1);"""
        code = '7-12'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_13(self):
        '''
        Basic behaviour for datasets with date type.
        '''
        text = """DS_r := timeshift(DS_1, -1);"""
        code = '7-13'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_14(self):
        '''
        Basic behaviour for datasets with date type.
        '''
        text = """DS_r := timeshift(DS_1, 0);"""
        code = '7-14'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_15(self):
        '''
        Basic behaviour for datasets with period type.
        '''
        text = """DS_r := timeshift(DS_1, 1);"""
        code = '7-15'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_16(self):
        '''
        Basic behaviour for datasets with period type.
        '''
        text = """DS_r := timeshift(DS_1, -1);"""
        code = '7-16'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_17(self):
        '''
        Basic behaviour for datasets with period type.
        '''
        text = """DS_r := timeshift(DS_1, 0);"""
        code = '7-17'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_18(self):
        '''
        Basic behaviour for datasets with period type.
        '''
        text = """DS_r := sum (DS_1 group all time_agg("A", Id_1));"""
        code = '7-18'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_19(self):
        '''
        Basic behaviour for datasets with date type.
        '''
        text = """DS_r := sum (DS_1 group all time_agg("A", Id_1));"""
        code = '7-19'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_20(self):
        '''
        Basic behaviour for group all with different durations (date, first)
        '''
        code = '7-20'
        number_inputs = 1
        references_names = ["1", "2", "3", "4", "5", "6"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_21(self):
        '''
        Basic behaviour for group all with different durations (date, last)
        '''
        code = '7-21'
        number_inputs = 1
        references_names = ["1", "2", "3", "4", "5", "6"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_22(self):
        '''
        Basic behaviour for dataset with different durations (date)
        '''
        code = '7-22'
        number_inputs = 1
        references_names = ["1", "2", "3", "4", "5", "6"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_23(self):
        '''
        Basic behaviour for dataset with different durations (time_period)
        '''
        code = '7-23'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_24(self):
        '''
        Dataset with calc on time_agg.
        '''
        code = '7-24'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_25(self):
        '''
        Semantic error on time_agg with periodIndTo = "D" on Time_period
        '''
        code = '7-25'
        number_inputs = 1
        message = "1-1-19-5"

        self.NewSemanticExceptionTest(text=None, code=code, number_inputs=number_inputs,
                                      exception_code=message)

    def test_26(self):
        '''
        Runtime Error on time_agg if any row has lower or equal duration than periodIndTo
        '''
        code = '7-26'
        number_inputs = 1
        message = "1-1-19-9"
        self.NewSemanticExceptionTest(text=None, code=code, number_inputs=number_inputs,
                                      exception_code=message)


class EmptyDatasetsTest(AdditionalHelper):
    """
    Group 14
    """

    classTest = 'Additional.EmptyDatasetsTest'

    maxDiff = None

    def test_1(self):
        '''

        '''
        code = '14-1'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_2(self):
        '''

        '''
        code = '14-2'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_3(self):
        '''

        '''
        code = '14-3'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_4(self):
        '''

        '''
        code = '14-4'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_5(self):
        '''

        '''
        code = '14-5'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_6(self):
        '''

        '''
        code = '14-6'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_7(self):
        '''

        '''
        code = '14-7'
        number_inputs = 3
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_8(self):
        '''

        '''
        code = '14-8'
        number_inputs = 3
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_9(self):
        '''

        '''
        code = '14-9'
        number_inputs = 3
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_10(self):
        '''

        '''
        code = '14-10'
        number_inputs = 3
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_11(self):
        '''

        '''
        code = '14-11'
        number_inputs = 3
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)

    def test_12(self):
        '''

        '''
        code = '14-12'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)


class DefinedOperatorsTest(AdditionalHelper):
    """
    Group 15
    """

    classTest = 'Additional.DefinedOperatorsTest'

    maxDiff = None

    def test_1(self):
        '''

        '''
        code = '15-1'
        self.BaseScalarTest(code=code, reference_value=3, text=None)

    def test_2(self):
        '''

        '''
        code = '15-2'
        self.BaseScalarTest(code=code, reference_value=2, text=None)

    def test_3(self):
        '''

        '''
        code = '15-3'
        number_inputs = 1
        references_names = ['DS_r1', 'DS_r2']

        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)


class DatesTest(AdditionalHelper):
    """
    Group 16
    """

    classTest = 'Additional.DatesTest'

    maxDiff = None

    def test_1(self):
        """

        """
        code = '16-1'
        number_inputs = 1
        references_names = ['DS_r']

        # with pytest.raises(Exception, match="cast .+? without providing a mask"):
        self.BaseTest(text=None, code=code, number_inputs=number_inputs,
                      references_names=references_names)
