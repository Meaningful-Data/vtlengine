import re
from pathlib import Path

import pytest

from tests.Helper import TestHelper


class UDOHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class UdoTest(UDOHelper):
    """
    Group 1
    """

    classTest = "udo.UdoTest"

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
        code = "1-1-1-1"
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
        code = "1-1-1-2"
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
        code = "1-1-1-3"
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
        """
        code = "1-1-1-4"
        number_inputs = 1
        message = "1-2-11"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )
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
        code = "1-1-1-5"
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
        code = "1-1-1-6"
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
        code = "1-1-1-7"
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
        code = "1-1-1-8"
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
        code = "1-1-1-9"
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
        code = "1-1-1-10"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        Expression: define operator identity(y component)
        returns component is
            y
        end operator;

        define operator is_above_05(x component)
        returns component is
            identity(x) > 0.5
        end operator;

        DS_r <- DS_1
        [calc
            column_3 := is_above_05(Me_1)
        ];
        Description: Bug on component.
        Git Branch: Cr-249
        """
        code = "1-1-1-12"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        USER DEFINED OPERATORS
        Dataset --> Dataset
        Status: OK
        Expression: define operator sum_except (x dataset, c component, e component, m component)
                        returns dataset is
                        x[calc Me_2 := sum(m over (partition except c order by e))]
                    end operator;
                    DS_A <- sum_except(DS_1, Id_1, Id_2, Me_1);
        Description: Component parameters of a UDO are resolved inside an analytic
                     partition clause, including the VTL 2.2 ``partition except`` form,
                     on both engines. Same data and result as Analytic GH_750_9.
        Goal: Check Result.
        """
        code = "1-1-1-13"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_286_1(self):
        """
        USER DEFINED OPERATORS
        Status: OK
        Description: This had the discrepance
        Git Branch: #286
        """
        code = "GL_286_1"
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
        code = "GL_286_2"
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
        code = "GL_286_3"
        number_inputs = 12
        references_names = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
        ]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_286_4(self):
        """
        USER DEFINED OPERATORS
        Status: OK
        Description:
        Git Branch: #286
        """
        code = "GL_286_4"
        number_inputs = 12
        references_names = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
        ]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_65(self):
        """
        UDO inside DPR
        Status: OK
        Description: Testing implementation UDO inside DPR
        Git Issue: #65
        """
        code = "GL_65"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_381(self):
        """
        Constant as Dataset input parameter
        """
        code = "GL_381"
        number_inputs = 0
        message = "1-3-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )
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
        code = "1-1-1-11"
        number_inputs = 1
        references_names = ["1"]
        vd_names = ["myGeoValueDomain"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names,
        )

    def test_GL_452_1(self):
        """
        Status: OK
        Description: UDO with scalar as input parameter
        Goal: Check Result.
        """
        code = "GL_452_1"
        number_inputs = 1
        message = "1-3-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_452_2(self):
        """
        Status: OK
        Description: UDO without return type defined, which is accepted (issue #1160), called
            with Components for its integer parameters
        Goal: Check Exception.
        """
        code = "GL_452_2"
        number_inputs = 1
        message = "1-3-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_452_3(self):
        """
        Status: OK
        Description: UDO with scalar as input parameter
        Goal: Check Result.
        """
        code = "GL_452_3"
        number_inputs = 0
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_452_4(self):
        """
        Status: OK
        Description: UDO with scalarDataset as input argument
        Goal: Check Result.
        """
        code = "GL_452_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            scalars={"sc_2": "4"},
        )

    def test_GL_452_5(self):
        """
        Status: OK
        Description: UDO with scalarDataset as input argument
        Goal: Check Result.
        """
        code = "GL_452_5"
        number_inputs = 2
        message = "1-3-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_442_1(self):
        """
        Status: OK
        Description: UDO with grouping by a component with a name that is a reserved word and a dot in the name
        Goal: Check Result.
        """
        code = "GL_442_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_442_2(self):
        """
        Constant as Dataset input parameter
        """
        code = "GL_442_2"
        number_inputs = 1
        message = "1-1-2-2"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_442_3(self):
        """
        Status: OK
        Description: complete example of issue 442
        Goal: Check Result.
        """
        code = "GL_442_3"
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_442_4(self):
        """
        Status: OK
        Description: complete example of issue 441
        Goal: Check Result.
        """
        code = "GL_442_4"
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_442_5(self):
        """
        Status: OK
        Description: there is no difference between component and 'component' (or "component"), the only diference is if component is a reserved word or not
        Goal: Check Result.
        """
        code = "GL_442_5"
        number_inputs = 2
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_473_1(self):
        """
        Status: OK
        Description: UDO with SDMX-CSV 1.0
        Goal: Check Result.
        """
        code = "GL_473_1"
        number_inputs = 1
        message = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_473_2(self):
        """
        Status: OK
        Description: UDO with SDMX-CSV 1.0
        Goal: Check Result.
        """
        code = "GL_473_2"
        number_inputs = 1
        message = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_474_1(self):
        """
        Status: OK
        Description: UDO with SDMX-CSV 1.0
        Goal: Check Result.
        """
        code = "GL_474_1"
        number_inputs = 1
        message = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_474_2(self):
        """
        Status: OK
        Description:
        Goal: Check Result.
        """
        code = "GL_474_2"
        number_inputs = 1
        message = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_474_3(self):
        """
        Status: OK
        Description:
        Goal: Check Result.
        """
        code = "GL_474_3"
        number_inputs = 1
        message = "1-2-3"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_475_1(self):
        """
        Status: OK
        Description: UDO with SDMX-CSV 1.0
        Goal: Check Result.
        """
        code = "GL_475_1"
        number_inputs = 1
        message = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_476_1(self):
        """
        Status: OK
        Description: The join body references ``A#CONF_STATUS`` / ``B#CONF_STATUS``
            in calc and keeps CONF_STATUS. Attributes are visible to the join body
            clauses (VTL 2.2) and the explicit keep retains CONF_STATUS.
        Goal: Check Result.
        """
        code = "GL_476_1"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1162_1(self):
        """
        Status: OK
        Description: An argument written as _ takes the default of its parameter, in any
            position and on operators over components and datasets
        Goal: Check Result.
        """
        code = "GH_1162_1"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1162_2(self):
        """
        Status: OK
        Description: An operator that calls itself is not supported
        Goal: Check Exception.
        """
        code = "GH_1162_2"
        number_inputs = 1
        message = "1-3-2-7"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1162_3(self):
        """
        Status: OK
        Description: Two operators that call each other are not supported
        Goal: Check Exception.
        """
        code = "GH_1162_3"
        number_inputs = 1
        message = "1-3-2-7"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1162_4(self):
        """
        Status: OK
        Description: A number with decimals passed to an integer parameter
        Goal: Check Exception.
        """
        code = "GH_1162_4"
        number_inputs = 1
        message = "1-2-18"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_94(self):
        """
        Status: OK
        Description: Test with analytic functions in UDO
        Goal: Check Result.
        """
        code = "GH_94"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_353(self):
        """
        Status: OK
        Description: Test with UDO using other UDO inside
        Goal: Check Result.
        """
        code = "GH_353"
        number_inputs = 0
        references_names = []
        error_message = "Invalid parameter type definition 'dattt' (line 1, column 32)."

        with pytest.raises(Exception, match=re.escape(error_message)):
            self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1161_1(self):
        """
        Status: OK
        Description: The body returns an integer, but the operator returns string
        Goal: Check Exception.
        """
        code = "GH_1161_1"
        number_inputs = 1
        message = "1-3-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1161_2(self):
        """
        Status: OK
        Description: The operator returns integer not null, and its body gives null
        Goal: Check Exception.
        """
        code = "GH_1161_2"
        number_inputs = 1
        message = "2-3-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1161_3(self):
        """
        Status: OK
        Description: The operator returns integer [value > 0], and its body gives -1
        Goal: Check Exception.
        """
        code = "GH_1161_3"
        number_inputs = 1
        message = "2-3-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1161_4(self):
        """
        Status: OK
        Description: The body returns a dataset without the Other_Id identifier the type names
        Goal: Check Exception.
        """
        code = "GH_1161_4"
        number_inputs = 1
        message = "1-3-1-11"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1161_5(self):
        """
        Status: OK
        Description: The default of an integer parameter is a string
        Goal: Check Exception.
        """
        code = "GH_1161_5"
        number_inputs = 1
        message = "1-3-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1161_6(self):
        """
        Status: OK
        Description: The call passes more arguments than the operator has parameters
        Goal: Check Exception.
        """
        code = "GH_1161_6"
        number_inputs = 1
        message = "1-3-1-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1161_7(self):
        """
        Status: OK
        Description: The results match the declared dataset structure and component type,
            and an integer default is accepted for a number parameter
        Goal: Check Result.
        """
        code = "GH_1161_7"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1160_1(self):
        """
        Status: OK
        Description: Operators without returns give the result of their expression: a Scalar,
            a Component and a Dataset, each with the type its expression gives
        Goal: Check Result.
        """
        code = "GH_1160_1"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1166_1(self):
        """
        Status: OK
        Description: Parameters and outputs with not null, conditions on value, allowed
            values and dataset types with _+ and _* components, all met by the call
        Goal: Check Result.
        """
        code = "GH_1166_1"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1166_2(self):
        """
        Status: OK
        Description: A null for a not null parameter
        Goal: Check Exception.
        """
        code = "GH_1166_2"
        number_inputs = 1
        message = "2-3-9"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
        )

    def test_GH_1166_3(self):
        """
        Status: OK
        Description: A value that breaks the condition of a parameter
        Goal: Check Exception.
        """
        code = "GH_1166_3"
        number_inputs = 1
        message = "2-3-9"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
        )

    def test_GH_1166_4(self):
        """
        Status: OK
        Description: A value outside the values a parameter allows
        Goal: Check Exception.
        """
        code = "GH_1166_4"
        number_inputs = 1
        message = "2-3-9"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
        )

    def test_GH_1166_5(self):
        """
        Status: OK
        Description: A null input scalar for a not null parameter
        Goal: Check Exception.
        """
        code = "GH_1166_5"
        number_inputs = 1
        message = "2-3-9"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            scalars={"SC_1": None},
        )

    def test_GH_1166_6(self):
        """
        Status: OK
        Description: A null in the data of a not null Component parameter
        Goal: Check Exception.
        """
        code = "GH_1166_6"
        number_inputs = 1
        message = "2-3-9"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
        )

    def test_GH_1166_7(self):
        """
        Status: OK
        Description: A Component that breaks the condition of its parameter, given outside a clause
        Goal: Check Exception.
        """
        code = "GH_1166_7"
        number_inputs = 1
        message = "2-3-9"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
        )

    def test_GH_1166_8(self):
        """
        Status: OK
        Description: A dataset with a Number Measure besides the one String Measure its type allows
        Goal: Check Exception.
        """
        code = "GH_1166_8"
        number_inputs = 1
        message = "1-3-1-14"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
        )

    def test_GH_1166_9(self):
        """
        Status: OK
        Description: A null in a Measure of the output that its type declares not null
        Goal: Check Exception.
        """
        code = "GH_1166_9"
        number_inputs = 1
        message = "2-3-9"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
        )

    def test_GH_1166_10(self):
        """
        Status: OK
        Description: A component that the dataset type names with another data type
        Goal: Check Exception.
        """
        code = "GH_1166_10"
        number_inputs = 1
        message = "1-3-1-12"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1166_11(self):
        """
        Status: OK
        Description: No component for the _+ of a dataset type
        Goal: Check Exception.
        """
        code = "GH_1166_11"
        number_inputs = 1
        message = "1-3-1-13"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1166_12(self):
        """
        Status: OK
        Description: Two components for the _ (exactly one) of a dataset type
        Goal: Check Exception.
        """
        code = "GH_1166_12"
        number_inputs = 1
        message = "1-3-1-14"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1166_13(self):
        """
        Status: OK
        Description: No component of the output for the _+ of the dataset type it returns
        Goal: Check Exception.
        """
        code = "GH_1166_13"
        number_inputs = 1
        message = "1-3-1-13"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1166_14(self):
        """
        Status: OK
        Description: A dataset with one String Measure for _, one Number Measure for _+ and no
            Boolean Measure for _*, and an output with one Number Measure for _
        Goal: Check Result.
        """
        code = "GH_1166_14"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
