from pathlib import Path

from tests.Helper import TestHelper


class DEMOHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class DEMO(DEMOHelper):
    """ """

    classTest = "md_demo.DEMO"

    def test_DEMO1(self):
        """ """
        code = "DEMO1"
        number_inputs = 4
        references_names = [
            "agg.DS2_tim",
            "agg.DS1_tim",
            "agg.val",
            "agg.exRate",
            "agg.DS1_conv",
            "agg.DS1_enr",
            "agg.DS1_fin",
            "aggr.numDPCouYear",
            "aggr.numYearCou",
            "aggr.numCouYear",
            "aggr.agg2",
            "aggr.agg1",
            "val.valResult_nonFiltered",
        ]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
