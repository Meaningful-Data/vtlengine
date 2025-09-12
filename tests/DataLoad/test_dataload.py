"""
tests.DataLoaddataloadtests.py
==============================

Description
-----------
Integration tests that check how CSV are loaded and its restrictions.

Summary
-------
- [1]: Base tests.
- [2]: Quotes on the types.
- [3]: Types of nulls.
- [4]: Types of boolean.
- [5]: Checking integers.
- [6]: Checking Time, Data and Time_period.
"""

from pathlib import Path

from tests.Helper import TestHelper


class DataLoadHelper(TestHelper):
    """ """

    # Path Selection.----------------------------------------------------------
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"


class DataLoadTest(DataLoadHelper):
    """
    Group DataLoad
    """

    classTest = "DataLoad.DataLoadTest"

    def test_1(self):
        """
        Status: OK
        Expression: DS_r := DS_1;
        Description: Data Load with all the types.
        Jira issue: VTLEN 423.
        Git Branch: csv_validation.
        Goal: Check Result.
        """
        code = "DataLoad-1"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Status: OK
        Expression: DS_r := DS_1;
        Description: Data Load with empty measures.
        Jira issue: VTLEN 423.
        Git Branch: csv_validation.
        Goal: Check Result.
        """
        code = "DataLoad-2"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Status: OK
        Expression: DS_r := DS_1;
        Description: Incorrect numbers.
        Jira issue: VTLEN 423.
        Git Branch: csv_validation.
        Goal: Check Exception.
        """
        code = "DataLoad-3"
        number_inputs = 1

        exception_code = "0-1-1-12"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_4(self):
        """
        Status: OK
        Expression: DS_r := DS_1;
        Description: Missing columns.
        Jira issue: VTLEN 423.
        Git Branch: csv_validation.
        Goal: Check Result.
        """
        code = "DataLoad-4"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        Status: OK
        Expression: DS_r := DS_1;
        Description: Missing columns and empty values.
        Jira issue: VTLEN 423.
        Git Branch: csv_validation.
        Goal: Check Result.
        """
        code = "DataLoad-5"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """ """
        code = "DataLoad-6"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """ """
        code = "DataLoad-7"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        Status: OK
        Expression: DS_r := DS_1;
        Description: Duplicated column.
        Jira issue: VTLEN 423.
        Git Branch: csv_validation.
        Goal: Check Exception.
        """
        code = "DataLoad-8"
        number_inputs = 1

        exception_code = "0-1-2-3"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_9(self):
        """
        Status: OK
        Expression: DS_r := DS_1;
        Description: bad value for date measure.
        Jira issue: VTLEN 423.
        Git Branch: csv_validation.
        Goal: Check Exception.
        """
        code = "DataLoad-9"
        number_inputs = 1

        message = ""
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    def test_10(self):
        """
        Status: OK
        Expression: DS_r := DS_1;
        Description: empty date as identifier.
        Jira issue: VTLEN 423.
        Git Branch: csv_validation.
        Goal: Check Exception.
        """
        code = "DataLoad-10"
        number_inputs = 1

        exception_code = "0-1-1-4"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_11(self):
        """
        Status: OK
        Expression: None
        Description:  Check validity for "," in a string.
        Git Branch: tests-41-csv-strings-could-not-be-loaded-at-dataset-validation-2
        Goal: Check Record.
        """
        code = "GL_41"
        number_inputs = 1
        string_to_compare = "Deshmoret e 4 Shkurtit, Godina nr. 6, Kati II"
        dataset_input = self.LoadInputs(code=code, number_inputs=number_inputs)["DS_1"]

        assert dataset_input.df["OBS_VALUE"][0] == string_to_compare

    def test_12(self):
        """
        Status: OK
        Description: Data Load, example bad quotes with commas.
        Git issue: 81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Git Branch: bug-81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Goal: Check Exception.
        """
        code = "GL_81-11"
        number_inputs = 1

        exception_code = "0-1-1-4"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_13(self):
        """
        Status: OK
        Description: Data Load, the same example correct writing.
        Git issue: 81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Git Branch: bug-81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Goal: Check Result.
        """
        code = "GL_81-12"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        Status: OK
        Description: Data Load.
        Git issue: 81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Git Branch: bug-81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Goal: Check Result.
        """
        code = "GL_81-13"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        Status: OK
        Description: Data Load,example bad quotes with commas.
        Git issue: 81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Git Branch: bug-81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Goal: Check Exception.
        """
        code = "GL_81-14"
        number_inputs = 1

        exception_code = "0-1-1-4"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_16(self):
        """
        Status: OK
        Description: Data Load, without headers.
        Git issue: 81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Git Branch: bug-81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Goal: Check Exception.
        """
        code = "GL_81-15"
        number_inputs = 1

        message = "The following identifiers Id_1 were not found , review file GL_81-15-1.csv"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    def test_17(self):
        """
        Status: OK
        Description: Data Load, without identifier(isNull false) in the headers.
        Git issue: 81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Git Branch: bug-81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Goal: Check Exception.
        """
        code = "GL_81-16"
        number_inputs = 1

        message = "The following identifiers VLD_T were not found , review file GL_81-16-1.csv"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    def test_18(self):
        """
        Status: OK
        Description: Data Load, without non nullable measure in the headers.
        Git issue: 81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Git Branch: bug-81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Goal: Check Exception.
        """
        code = "GL_81-17"
        number_inputs = 1

        exception_code = "0-1-1-10"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_19(self):
        """
        Status: OK
        Description: Data Load with value in quotes in the identifiers.
        Git issue: 81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Git Branch: bug-81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Goal: Check Result.
        """
        code = "GL_81-18"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        """
        Status: OK
        Description: Data Load, with ñ in a latin-1(ISO-8859-1) encoding.
        Git issue: 81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Git Branch: bug-81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Goal: Check Exception.
        """
        # code = "GL_81-19"
        # number_inputs = 1
        # message = "0-1-2-5"
        # self.DataLoadExceptionTest(code=code, number_inputs=number_inputs,
        #                            exception_code=message)

    def test_21(self):
        """
        Status: OK
        Description: Data Load, without ñ in a latin-1(ISO-8859-1) encoding.
        Git issue: 81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Git Branch: bug-81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Goal: Check Exception.
        """
        code = "GL_81-20"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_22(self):
        """
        Status: OK
        Description: Data Load, with utf-16 encoding.
        Git issue: 81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Git Branch: bug-81-triple-doble-quote-commas-data-loading-and-intermediate-results.
        Goal: Check Exception.
        """
        # code = "GL_81-21"
        # number_inputs = 1
        # message = "ERROR: line contains NUL"
        # TODO: Check the dialect on the Dataload.
        # self.DataLoadExceptionTest(code=code, number_inputs=number_inputs,
        #                            exception_message=message)
        assert True

    # Quotes on the types
    def test_23(self):
        """
        Status: OK
        Description: Data Load, with "" is omitted, but with \"\"\" fails.
        Git issue: 91-complete-data-load.
        Git Branch: 91-complete-data-load.
        Goal: Check Exception.
        """
        code = "GL_91-22"
        number_inputs = 1

        exception_code = "0-1-1-12"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_24(self):
        """
        Status: OK
        Description: Data Load, with spaces. Left spaces for integer,number
                    duration and string are allowed.
        Git issue: 91-complete-data-load.
        Git Branch: 91-complete-data-load.
        Goal: Check Exception.
        """
        code = "GL_91-23"
        number_inputs = 1

        exception_code = "0-1-1-12"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_25(self):
        """
        Status: OK
        Description: Data Load, with spaces. Right spaces for integer,number
                    duration and string are allowed.
        Git issue: 91-complete-data-load.
        Git Branch: 91-complete-data-load.
        Goal: Check Exception.
        """
        code = "GL_91-24"
        number_inputs = 1

        exception_code = "0-1-1-12"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_26(self):
        """
        Status: OK
        Description: check if the spaces are represented for the types allowed.
        Git Branch: 91-complete-data-load
        Goal: Check Record.
        """
        code = "GL_91-25"
        number_inputs = 1
        references_names = ["DS_r"]

        self.DataLoadTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_27(self):
        """
        Status: OK
        Expression: None
        Description:  Any string is allowed for duration.
        Git Branch: 91-complete-data-load
        Goal: Check Exception.
        """
        code = "GL_91-26"
        number_inputs = 1

        exception_code = "0-1-1-12"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_28(self):
        """
        Status:
        Description: Null identifiers are allowed.
        Git Branch: 91-complete-data-load.
        Goal: Check Record.
        """
        code = "GL_91-27"
        number_inputs = 1
        exception_code = "0-1-1-4"

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_29(self):
        """
        Status: OK
        Description:  <space> and "<space>" is not allowed as null.
        Git Branch: 91-complete-data-load
        Goal: Check Exception.
        """
        code = "GL_91-29"
        number_inputs = 1

        exception_code = "0-1-1-12"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_30(self):
        """
        Status: OK
        Description:  None is not allowed as null.
        Git Branch: 91-complete-data-load
        Goal: Check Exception.
        """
        code = "GL_91-30"
        number_inputs = 1

        exception_code = "0-1-1-12"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    # Types of boolean
    def test_31(self):
        """
        Status: OK
        Description:  FALSE, False and false are allowed.
        Git Branch: 91-complete-data-load
        Goal: Check Record.
        """
        code = "GL_91-31"
        number_inputs = 1

        references_names = ["DS_r"]

        self.DataLoadTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_32(self):
        """
        Status: OK
        Description:  0 and \"\"\" are not allowed.
        Git Branch: 91-complete-data-load
        Goal: Check Exception.
        """
        code = "GL_91-32"
        number_inputs = 1

        self.DataLoadTest(code=code, number_inputs=number_inputs)

    def test_33(self):
        """
        Status:
        Description:  Problem, when isNull==True for all the components on the
            compute of getMandatory components.
            Error: 'NoneType' object is not iterable .
        Git Branch: 91-complete-data-load
        Goal: Check Exception.
        """
        code = "GL_91-33"
        number_inputs = 1

        exception_code = "0-1-1-4"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    # Checking integers
    def test_34(self):
        """
        Status: OK
        Description:  number.decimal(1-9) is not allowed.
        Git Branch: 91-complete-data-load
        Goal: Check Exception.
        """
        code = "GL_91-34"
        number_inputs = 1

        message = "On Dataset DS_1 loading: not possible to cast column Me_1 to Integer"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    def test_35(self):
        """
        Status: OK
        Description:  2.0, 0.2e3 and 2000e-3  are allowed.
        Git Branch: 91-complete-data-load
        Goal: Check Exception.
        """
        code = "GL_91-35"
        number_inputs = 1

        references_names = ["DS_r"]

        self.DataLoadTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_36(self):
        """
        Status: OK
        Description:  0.211e2 and 20e-3 aren't allowed.
        Git Branch: 91-complete-data-load
        Goal: Check Exception.
        """
        code = "GL_91-36"
        number_inputs = 1

        message = "On Dataset DS_1 loading: not possible to cast column Me_1 to Integer"

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    # Checking Time, Data and Time_period
    # Time and Time period
    def test_37(self):
        """
        Status: OK
        Description: Wrong letters are not allowed.
        Git Branch: 91-complete-data-load
        Goal: Check Exception.
        """
        code = "GL_91-37"
        number_inputs = 1

        message = "On Dataset DS_1 loading: not possible to cast column Me_1 to Time"

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    def test_38(self):
        """
        Status: OK
        Description:  M01==M1
        Git Branch: 91-complete-data-load
        Goal: Check Exception.
        """
        code = "GL_91-38"
        number_inputs = 1

        references_names = ["DS_r"]

        self.DataLoadTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_39(self):
        """
        Status: OK
        Description: Fail when M000 and M13+.
        Git Branch: 91-complete-data-load
        Goal: Check Exception.
        """
        code = "GL_91-39"
        number_inputs = 1

        message = "On Dataset DS_1 loading: not possible to cast column Me_1 to Time"

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    # Time
    def test_40(self):
        """
        Status: OK
        Description:  2010Q2/2010M12
        Git Branch: 91-complete-data-load
        Goal: Check Exception.
        """
        code = "GL_91-40"
        number_inputs = 1

        message = "On Dataset DS_1 loading: not possible to cast column Me_1 to Time"

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    # Date
    def test_41(self):
        """
        Status: OK
        Description:  2014/12/31 is not allowed.
        Git Branch: 91-complete-data-load
        Goal: Check Exception.
        """
        code = "GL_91-41"
        number_inputs = 1

        exception_code = "0-1-1-12"

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_42(self):
        """
        Status: OK
        Description: non-existent dates(30 February,...) are not allowed.
        Git Branch: 91-complete-data-load
        Goal: Check Exception.
        """
        code = "GL_91-42"
        number_inputs = 1

        exception_code = "0-1-1-12"

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    # Checking Identifiers without identifiers and nullable identifiers.
    def test_43(self):
        """
        Status: OK
        Description: Identifier with isNull: True.
        Git Branch: bug-gl-100_fix-load-input-datastructure-for-not-allow-null-identifiers
        Goal: Check Exception.
        """
        code = "GL_100_1"
        number_inputs = 1

        exception_code = "0-1-1-4"
        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_45(self):
        """
        Description: Fix dataload for type duration.
        Git Branch: bug-99-fix-dataload-for-type-duration.
        Goal: Interpreter results.
        """
        code = "GL_91-26"
        number_inputs = 1

        message = "On Dataset DS_1 loading: not possible to cast column Me_1 to Duration"

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    def test_46(self):
        """
        Description: Dataset without identifiers can only have one row
        Git Branch: feat-199-DWI-loading
        Goal: Interpreter results.
        """
        code = "GL_91-43"
        number_inputs = 1

        message = "Datasets without identifiers must have 0 or 1 datapoints"

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    def test_47(self):
        """
        Status: OK
        Description: Datasets without identifiers loading correctly.
        Git Branch: feat-199-DWI-loading
        Goal: Check Result.
        """
        code = "GL_91-44"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_48(self):
        """
        Status: OK
        Description: Loading iso 8601 masks to perform the new duration type with 'month to day' op.
        Git Branch: 55-check-duration-type
        Goal: Check Result
        """
        code = "DataLoad-11"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_49(self):
        """
        Status: OK
        Description: Loading iso 8601 masks to perform the new duration type with 'year to day' op.
        Git Branch: 55-check-duration-type
        Goal: Check Result
        """
        code = "DataLoad-12"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_50(self):
        """
        Status: OK
        Description: Loading iso 8601 masks to perform the new duration type with error input.
        Git Branch: 55-check-duration-type
        Goal: Check Result
        """
        code = "DataLoad-13"
        number_inputs = 1

        message = "On Dataset DS_1 loading: not possible to cast column Me_1 to Duration."

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    def test_GL_483_1(self):
        """
        Status: OK
        Description: load values that before were loaded as null.
        Goal: Interpreter results.
        """
        code = "GL_483_1"
        number_inputs = 1

        references_names = ["1"]
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # TODO: Removed because dataload will be with pysdmx
    # def test_GL_492_1(self):
    #     """
    #     Status: OK
    #     Description:  SDMX-CSV 1.0 loading
    #     Git Branch: 492-accept-sdmx-csv-1-0-and-2-0-on-dataset-load
    #     Goal: Check Record.
    #     """
    #     code = "GL_492-1"
    #     number_inputs = 1
    #     references_names = ["DS_r"]
    #
    #     self.DataLoadTest(code=code, number_inputs=number_inputs, references_names=references_names)
    #
    # def test_GL_492_2(self):
    #     """
    #     Status: OK
    #     Description:  SDMX-CSV 2.0 loading
    #     Git Branch: 492-accept-sdmx-csv-1-0-and-2-0-on-dataset-load
    #     Goal: Check Record.
    #     """
    #     code = "GL_492-2"
    #     number_inputs = 1
    #     references_names = ["DS_r"]
    #
    #     self.DataLoadTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_infer_keys_1(self):
        """ """
        code = "IK-1"
        number_inputs = 1
        message = "Invalid key on role field: Identfier. Did you mean Identifier?."

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    def test_infer_keys_2(self):
        """ """
        code = "IK-2"
        number_inputs = 1
        message = "Invalid key on role field: Masure. Did you mean Measure?."

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    def test_infer_keys_3(self):
        """ """
        code = "IK-3"
        number_inputs = 1
        message = "Invalid key on data_type field: Numver. Did you mean Number?."

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    def test_infer_keys_4(self):
        """ """
        code = "IK-4"
        number_inputs = 1
        message = "Invalid key on data_type field: boolean. Did you mean Boolean?."

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    def test_infer_keys_5(self):
        """ """
        code = "IK-5"
        number_inputs = 1
        message = "Invalid key on data_type field: TimePeriod. Did you mean Time_Period?."

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    def test_infer_keys_6(self):
        """ """
        code = "IK-6"
        number_inputs = 1
        message = "Invalid key on data_type field: TimPerod. Did you mean Time_Period?."

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    def test_infer_keys_7(self):
        """ """
        code = "IK-7"
        number_inputs = 1
        message = "Invalid key on data_type field: jbhfae."

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )
