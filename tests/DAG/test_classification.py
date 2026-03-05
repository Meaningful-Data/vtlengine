from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pytest

from vtlengine.API import create_ast
from vtlengine.AST.DAG import DAGAnalyzer

data_path = Path(__file__).parent / "data"


@dataclass
class Classification:
    datasets: List[str] = field(default_factory=list)
    scalars: List[str] = field(default_factory=list)
    dataset_or_scalar: List[str] = field(default_factory=list)
    component_or_scalar: List[str] = field(default_factory=list)
    vtl: Optional[str] = None


# Tests 1-11: vtl=None → read from VTL files (complex multi-statement scripts).
# Tests 12+: vtl is inline.
CASES: dict[str, Classification] = {
    # --- File-based tests (complex multi-statement scripts) ---
    "1": Classification(),
    "2": Classification(dataset_or_scalar=["A"]),
    "3": Classification(
        datasets=["A", "A2"],
        component_or_scalar=["var1", "var3", "varF", "varRel", "varRel2"],
    ),
    "4": Classification(datasets=["DS_1", "DS_2"]),
    "5": Classification(
        datasets=["DSD_AGR", "DSD_POP"],
        component_or_scalar=["AGE", "MEASURE", "SEX", "TIME_HORIZ", "UNIT_MEASURE"],
    ),
    "6": Classification(
        datasets=["BIS_LOC_STATS", "DS1", "DS2", "DS3"],
        component_or_scalar=[
            "CURRENCY",
            "CURRENCY_DENOM",
            "EXCHANGE_RATE",
            "EXR_SUFFIX",
            "EXR_TYPE",
            "FREQ",
            "OBS_VALUE",
        ],
    ),
    "7": Classification(
        datasets=[
            "ANCRDT_ACCNTNG_C",
            "ANCRDT_ACCNTNG_C_Z",
            "ANCRDT_ENTTY",
            "ANCRDT_ENTTY_DFLT_C",
            "ANCRDT_ENTTY_DFLT_C_T1",
            "ANCRDT_ENTTY_INSTRMNT_C",
            "ANCRDT_ENTTY_RSK_C",
            "ANCRDT_FNNCL_C",
            "ANCRDT_FNNCL_C_T1",
            "ANCRDT_INSTRMNT_C",
            "ANCRDT_INSTRMNT_C_T1",
            "ANCRDT_INSTRMNT_PRTCTN_RCVD_C",
            "ANCRDT_JNT_LBLTS_C",
            "ANCRDT_PRTCTN_RCVD_C",
            "ANCRDT_PRTCTN_RCVD_C_T1",
        ],
        component_or_scalar=[
            "ACCMLTD_WRTFFS",
            "ANCRDT_DRGTN_QRTR_CR_OA",
            "CC0010",
            "CNTRY",
            "CRDTR",
            "CRDTR_CD",
            "DBTR_CD",
            "DFLT_STTS",
            "DT_BRTH",
            "DT_INCPTN",
            "DT_RFRNC",
            "ENTTY_RIAD_CD",
            "ENTTY_RL",
            "FRGN_BRNCH",
            "HD_OFFC_UNDRT_CD",
            "HD_OFFC_UNDRT_CNTRY",
            "HD_QRTR_CD_CRDTR",
            "HD_QRTR_CD_DBTR",
            "IMMDT_PRNT_UNDRT_CD",
            "INSTTTNL_SCTR",
            "INSTTTNL_SCTR_DTL",
            "IS_PRTCTN_PRVDR",
            "LGL_FRM",
            "OBSRVD_AGNT_CD",
            "OFF_BLNC_SHT_AMNT",
            "OTHR_TYP_ENTTY",
            "OTSTNDNG_NMNL_AMNT",
            "PRTCTN_ALLCTD_VL",
            "PRTCTN_PRVDR_CD",
            "RCGNTN_STTS",
            "RCRS",
            "SPFUND",
            "SRVCR",
            "SSMSIGNIFICANCE",
            "THRD_PRTY_PRRTY_CLMS",
            "TRD_RCVBL_NN_RCRS",
            "TTL_NMBR_DBTRS",
            "TTL_NMBR_DFLT_DBTRS",
            "TYP_INSTRMNT",
            "TYP_PRTCTN",
            "TYP_SCRTSTN",
            "ULTMT_PRNT_UNDRT_CD",
        ],
    ),
    "8": Classification(
        datasets=[
            "ANCRDT_ACCNTNG_C",
            "ANCRDT_ACCNTNG_C_T3",
            "ANCRDT_ENTTY",
            "ANCRDT_ENTTY_DFLT_C",
            "ANCRDT_ENTTY_INSTRMNT_C",
            "ANCRDT_FNNCL_C",
            "ANCRDT_INSTRMNT_C",
            "ANCRDT_INSTRMNT_C_T1",
            "ANCRDT_INSTRMNT_C_T2",
            "ANCRDT_INSTRMNT_C_T3",
        ],
        component_or_scalar=[
            "ENTTY_RIAD_CD",
            "ENTTY_RL",
            "HD_OFFC_UNDRT_CD",
            "LGL_ENTTY_CD",
            "OBSRVD_AGNT_CD",
        ],
    ),
    "9": Classification(
        datasets=[
            "Income_PT",
            "Inflation_PT",
            "Inflation_divisors_Q",
            "Oferta_PT_2025_Q1",
            "Vendas_PT_2025_Q1",
        ],
        component_or_scalar=[
            "coefficient",
            "coefficient_cq",
            "coefficient_inv",
            "coefficient_lc",
            "coefficient_lcq",
            "coefficient_q",
            "county",
            "divisor",
            "estado",
            "income",
            "period_label",
            "regiao",
            "value",
            "var",
            "year_str",
        ],
    ),
    "10": Classification(
        datasets=["BOP"],
        component_or_scalar=[
            "ACCOUNTING_ENTRY",
            "ADJUSTMENT",
            "COMP_METHOD",
            "COUNTERPART_SECTOR",
            "CURRENCY_DENOM",
            "FLOW_STOCK_ENTRY",
            "FREQ",
            "FUNCTIONAL_CAT",
            "INSTR_ASSET",
            "INT_ACC_ITEM",
            "MATURITY",
            "REF_SECTOR",
            "VALUATION",
            "imbalance",
        ],
    ),
    "11": Classification(),
    # --- Inline tests ---
    # Calc with external component/scalar
    "12": Classification(
        vtl="DS_r <- DS_1[calc Me_2 := Me_1 * SC_1];",
        datasets=["DS_1"],
        component_or_scalar=["Me_1", "SC_1"],
    ),
    # Scalar chain with UDO
    "13": Classification(
        vtl="SC_a := SC_1 + SC_2;\nDS_r <- DS_1[calc Me_2 := Me_1 + SC_a];",
        datasets=["DS_1"],
        scalars=["SC_1", "SC_2"],
        component_or_scalar=["Me_1"],
    ),
    # Dual binary op
    "14": Classification(
        vtl="DS_r <- DS_1 + DS_2;",
        dataset_or_scalar=["DS_1", "DS_2"],
    ),
    # Scalar chain feeding calc
    "15": Classification(
        vtl="SC_r := 10;\nDS_r <- DS_1[calc Me_2 := Me_1 + SC_r];",
        datasets=["DS_1"],
        component_or_scalar=["Me_1"],
    ),
    # Mixed dataset_or_scalar + scalar chain
    "16": Classification(
        vtl="DS_r <- DS_1 + SC_1;\nSC_r := SC_1 * 2;",
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    # Membership
    "17": Classification(vtl="DS_r := DS_1#Me_1;", datasets=["DS_1"]),
    # Set operators (dataset-only)
    "18": Classification(vtl="DS_r <- union(DS_1, DS_2);", datasets=["DS_1", "DS_2"]),
    "19": Classification(vtl="DS_r <- intersect(DS_1, DS_2);", datasets=["DS_1", "DS_2"]),
    "20": Classification(vtl="DS_r <- setdiff(DS_1, DS_2);", datasets=["DS_1", "DS_2"]),
    # If-then-else (dual)
    "21": Classification(
        vtl="DS_r := if DS_1 then DS_2 else DS_3;",
        dataset_or_scalar=["DS_1", "DS_2", "DS_3"],
    ),
    "22": Classification(
        vtl="SC_r := if true then SC_1 else SC_2;",
        dataset_or_scalar=["SC_1", "SC_2"],
    ),
    # Comparison / logical (dual)
    "23": Classification(vtl="DS_r := DS_1 > DS_2;", dataset_or_scalar=["DS_1", "DS_2"]),
    "24": Classification(vtl="DS_r := DS_1 and DS_2;", dataset_or_scalar=["DS_1", "DS_2"]),
    "25": Classification(vtl="DS_r := not DS_1;", dataset_or_scalar=["DS_1"]),
    # Numeric (dual)
    "26": Classification(vtl="DS_r := abs(DS_1);", dataset_or_scalar=["DS_1"]),
    "27": Classification(vtl="SC_r := abs(SC_1);", dataset_or_scalar=["SC_1"]),
    # Aggregation (dataset-only)
    "28": Classification(vtl="DS_r := sum(DS_1);", datasets=["DS_1"]),
    # Clause operators
    "29": Classification(
        vtl="DS_r <- DS_1[filter Me_1 > 10];",
        datasets=["DS_1"],
        component_or_scalar=["Me_1"],
    ),
    "30": Classification(vtl="DS_r <- DS_1[keep Me_1];", datasets=["DS_1"]),
    "31": Classification(vtl="DS_r <- DS_1[drop Me_1];", datasets=["DS_1"]),
    "32": Classification(vtl="DS_r <- DS_1[rename Me_1 to Me_2];", datasets=["DS_1"]),
    "33": Classification(
        vtl="DS_r := DS_1[sub Id_1 = 1];",
        datasets=["DS_1"],
        component_or_scalar=["Id_1"],
    ),
    # Parameterized (dual)
    "34": Classification(vtl="DS_r := round(DS_1, 2);", dataset_or_scalar=["DS_1"]),
    # Multi-statement with intermediates
    "35": Classification(
        vtl="DS_A := DS_1 + DS_2;\nDS_B := DS_1 * DS_3;\nDS_r := DS_A + DS_B;",
        dataset_or_scalar=["DS_1", "DS_2", "DS_3"],
    ),
    # Scalar chain propagation
    "36": Classification(
        vtl="SC_a := 10;\nSC_b := SC_a + SC_1;\nDS_r <- DS_1[calc Me_2 := Me_1 + SC_b];",
        datasets=["DS_1"],
        scalars=["SC_1"],
        component_or_scalar=["Me_1"],
    ),
    # Join (dataset-only)
    "37": Classification(vtl="DS_r := inner_join(DS_1, DS_2);", datasets=["DS_1", "DS_2"]),
    # Dual unary
    "38": Classification(vtl="DS_r := isnull(DS_1);", dataset_or_scalar=["DS_1"]),
    "39": Classification(vtl="DS_r := -DS_1;", dataset_or_scalar=["DS_1"]),
    # Calc with multiple external refs
    "40": Classification(
        vtl="DS_r <- DS_1[calc Me_2 := Me_1 + SC_1, Me_3 := Me_1 * SC_2];",
        datasets=["DS_1"],
        component_or_scalar=["Me_1", "SC_1", "SC_2"],
    ),
    # UDO with typed parameters
    "41": Classification(
        vtl=(
            "define operator my_op (ds dataset, sc number)\n"
            "  returns dataset is\n"
            "    ds * sc\n"
            "end operator;\n\n"
            "DS_r := my_op(DS_1, SC_1);"
        ),
        datasets=["DS_1"],
        scalars=["SC_1"],
    ),
    # Time operators (dataset-only)
    "42": Classification(vtl="DS_r := flow_to_stock(DS_1);", datasets=["DS_1"]),
    "43": Classification(vtl="DS_r := stock_to_flow(DS_1);", datasets=["DS_1"]),
    "44": Classification(vtl="DS_r := exists_in(DS_1, DS_2, all);", datasets=["DS_1", "DS_2"]),
    "45": Classification(vtl="DS_r := timeshift(DS_1, 1);", datasets=["DS_1"]),
    # --- Dual BinOp operators ---
    "46": Classification(vtl="DS_r := DS_1 / DS_2;", dataset_or_scalar=["DS_1", "DS_2"]),
    "47": Classification(vtl="DS_r := mod(DS_1, DS_2);", dataset_or_scalar=["DS_1", "DS_2"]),
    "48": Classification(vtl="DS_r := DS_1 || DS_2;", dataset_or_scalar=["DS_1", "DS_2"]),
    "49": Classification(vtl="DS_r := DS_1 or DS_2;", dataset_or_scalar=["DS_1", "DS_2"]),
    "50": Classification(vtl="DS_r := DS_1 xor DS_2;", dataset_or_scalar=["DS_1", "DS_2"]),
    "51": Classification(vtl="DS_r := DS_1 = DS_2;", dataset_or_scalar=["DS_1", "DS_2"]),
    "52": Classification(vtl="DS_r := DS_1 <> DS_2;", dataset_or_scalar=["DS_1", "DS_2"]),
    "53": Classification(vtl="DS_r := DS_1 >= DS_2;", dataset_or_scalar=["DS_1", "DS_2"]),
    "54": Classification(vtl="DS_r := DS_1 < DS_2;", dataset_or_scalar=["DS_1", "DS_2"]),
    "55": Classification(vtl="DS_r := DS_1 <= DS_2;", dataset_or_scalar=["DS_1", "DS_2"]),
    "56": Classification(vtl="DS_r := DS_1 in {1, 2, 3};", dataset_or_scalar=["DS_1"]),
    "57": Classification(vtl="DS_r := DS_1 not_in {1, 2, 3};", dataset_or_scalar=["DS_1"]),
    "58": Classification(
        vtl='DS_r := match_characters(DS_1, "[a-z]+");',
        dataset_or_scalar=["DS_1"],
    ),
    "59": Classification(vtl="DS_r := nvl(DS_1, 0);", dataset_or_scalar=["DS_1"]),
    # --- Dual UnaryOp operators ---
    "60": Classification(vtl="DS_r := exp(DS_1);", dataset_or_scalar=["DS_1"]),
    "61": Classification(vtl="DS_r := ln(DS_1);", dataset_or_scalar=["DS_1"]),
    "62": Classification(vtl="DS_r := sqrt(DS_1);", dataset_or_scalar=["DS_1"]),
    "63": Classification(vtl="DS_r := ceil(DS_1);", dataset_or_scalar=["DS_1"]),
    "64": Classification(vtl="DS_r := floor(DS_1);", dataset_or_scalar=["DS_1"]),
    "65": Classification(vtl="DS_r := +DS_1;", dataset_or_scalar=["DS_1"]),
    "66": Classification(vtl="DS_r := length(DS_1);", dataset_or_scalar=["DS_1"]),
    "67": Classification(vtl="DS_r := upper(DS_1);", dataset_or_scalar=["DS_1"]),
    "68": Classification(vtl="DS_r := lower(DS_1);", dataset_or_scalar=["DS_1"]),
    "69": Classification(vtl="DS_r := trim(DS_1);", dataset_or_scalar=["DS_1"]),
    "70": Classification(vtl="DS_r := ltrim(DS_1);", dataset_or_scalar=["DS_1"]),
    "71": Classification(vtl="DS_r := rtrim(DS_1);", dataset_or_scalar=["DS_1"]),
    # --- Dual ParamOp operators ---
    "72": Classification(vtl="DS_r := trunc(DS_1, 2);", dataset_or_scalar=["DS_1"]),
    "73": Classification(vtl="DS_r := power(DS_1, 2);", dataset_or_scalar=["DS_1"]),
    "74": Classification(vtl="DS_r := log(DS_1, 10);", dataset_or_scalar=["DS_1"]),
    "75": Classification(vtl="DS_r := substr(DS_1, 1, 3);", dataset_or_scalar=["DS_1"]),
    "76": Classification(vtl='DS_r := replace(DS_1, "a", "b");', dataset_or_scalar=["DS_1"]),
    "77": Classification(vtl='DS_r := instr(DS_1, "a");', dataset_or_scalar=["DS_1"]),
    "78": Classification(vtl="DS_r := cast(DS_1, integer);", dataset_or_scalar=["DS_1"]),
    # --- Dual MulOp / conditional ---
    "79": Classification(vtl="DS_r := between(DS_1, 1, 10);", dataset_or_scalar=["DS_1"]),
    "80": Classification(
        vtl="DS_r := case when DS_1 > 0 then DS_2 else DS_3;",
        dataset_or_scalar=["DS_1", "DS_2", "DS_3"],
    ),
    # --- Dual time operators ---
    "81": Classification(
        vtl="DS_r := datediff(DS_1, DS_2);",
        dataset_or_scalar=["DS_1", "DS_2"],
    ),
    "82": Classification(vtl='DS_r := dateadd(DS_1, 1, "M");', dataset_or_scalar=["DS_1"]),
    "83": Classification(vtl="DS_r := getyear(DS_1);", dataset_or_scalar=["DS_1"]),
    "84": Classification(vtl="DS_r := getmonth(DS_1);", dataset_or_scalar=["DS_1"]),
    "85": Classification(vtl="DS_r := dayofmonth(DS_1);", dataset_or_scalar=["DS_1"]),
    "86": Classification(vtl="DS_r := dayofyear(DS_1);", dataset_or_scalar=["DS_1"]),
    # --- Dataset-only operators ---
    "87": Classification(vtl="DS_r := symdiff(DS_1, DS_2);", datasets=["DS_1", "DS_2"]),
    "88": Classification(vtl="DS_r := left_join(DS_1, DS_2);", datasets=["DS_1", "DS_2"]),
    "89": Classification(vtl="DS_r := full_join(DS_1, DS_2);", datasets=["DS_1", "DS_2"]),
    "90": Classification(
        vtl="DS_r := cross_join(DS_1 as d1, DS_2 as d2);",
        datasets=["DS_1", "DS_2"],
    ),
    "91": Classification(vtl="DS_r := count(DS_1);", datasets=["DS_1"]),
    "92": Classification(vtl="DS_r := min(DS_1);", datasets=["DS_1"]),
    "93": Classification(vtl="DS_r := max(DS_1);", datasets=["DS_1"]),
    "94": Classification(vtl="DS_r := avg(DS_1);", datasets=["DS_1"]),
    "95": Classification(vtl="DS_r := median(DS_1);", datasets=["DS_1"]),
    "96": Classification(vtl="DS_r := stddev_pop(DS_1);", datasets=["DS_1"]),
    "97": Classification(vtl="DS_r := stddev_samp(DS_1);", datasets=["DS_1"]),
    "98": Classification(vtl="DS_r := var_pop(DS_1);", datasets=["DS_1"]),
    "99": Classification(vtl="DS_r := var_samp(DS_1);", datasets=["DS_1"]),
    "100": Classification(vtl="DS_r := fill_time_series(DS_1, all);", datasets=["DS_1"]),
    "101": Classification(vtl="DS_r := period_indicator(DS_1);", datasets=["DS_1"]),
    # --- Clause operators ---
    "102": Classification(
        vtl="DS_r := DS_1[aggr Me_1 := sum(Me_2) group by Id_1];",
        datasets=["DS_1"],
        component_or_scalar=["Me_2"],
    ),
    "103": Classification(vtl="DS_r := DS_1[pivot Id_1, Me_1];", datasets=["DS_1"]),
    "104": Classification(vtl="DS_r := DS_1[unpivot Id_1, Me_1];", datasets=["DS_1"]),
    # --- Analytic operators (operands are always components, never scalars) ---
    "105": Classification(
        vtl="DS_r := DS_1[calc Me_2 := first_value(Me_1 over (order by Id_1))];",
        datasets=["DS_1"],
    ),
    "106": Classification(
        vtl="DS_r := DS_1[calc Me_2 := last_value(Me_1 over (order by Id_1))];",
        datasets=["DS_1"],
    ),
    "107": Classification(
        vtl="DS_r := DS_1[calc Me_2 := lag(Me_1, 1 over (order by Id_1))];",
        datasets=["DS_1"],
    ),
    "108": Classification(
        vtl="DS_r := DS_1[calc Me_2 := lead(Me_1, 1 over (order by Id_1))];",
        datasets=["DS_1"],
    ),
    "109": Classification(
        vtl="DS_r := DS_1[calc Me_2 := rank(over (order by Id_1))];",
        datasets=["DS_1"],
    ),
    "110": Classification(
        vtl="DS_r := DS_1[calc Me_2 := ratio_to_report(Me_1 over (partition by Id_1))];",
        datasets=["DS_1"],
    ),
    # --- Join clause operators ---
    "111": Classification(
        vtl="DS_r := inner_join(DS_1, DS_2 calc Me_2 := Me_1 + SC_1);",
        datasets=["DS_1", "DS_2"],
        component_or_scalar=["Me_1", "SC_1"],
    ),
    "112": Classification(
        vtl="DS_r := inner_join(DS_1, DS_2 filter Me_1 > 10);",
        datasets=["DS_1", "DS_2"],
        component_or_scalar=["Me_1"],
    ),
    "113": Classification(
        vtl="DS_r := inner_join(DS_1, DS_2 keep Me_1);",
        datasets=["DS_1", "DS_2"],
    ),
    "114": Classification(
        vtl="DS_r := inner_join(DS_1, DS_2 rename Me_1 to Me_2);",
        datasets=["DS_1", "DS_2"],
    ),
    "115": Classification(
        vtl="DS_r := inner_join(DS_1, DS_2 using Id_1);",
        datasets=["DS_1", "DS_2"],
    ),
    "116": Classification(
        vtl="DS_r := inner_join(DS_1, DS_2, DS_3);",
        datasets=["DS_1", "DS_2", "DS_3"],
    ),
    "117": Classification(
        vtl="DS_r := inner_join(DS_1, DS_2 using Id_1 calc Me_2 := Me_1 + SC_1);",
        datasets=["DS_1", "DS_2"],
        component_or_scalar=["Me_1", "SC_1"],
    ),
    "118": Classification(
        vtl="DS_r := inner_join(DS_1, DS_2 aggr Me_1 := sum(Me_2) group by Id_1);",
        datasets=["DS_1", "DS_2"],
        component_or_scalar=["Me_2"],
    ),
    "119": Classification(
        vtl="DS_r := left_join(DS_1, DS_2 calc Me_2 := Me_1 * 2);",
        datasets=["DS_1", "DS_2"],
        component_or_scalar=["Me_1"],
    ),
    "120": Classification(
        vtl="DS_r := full_join(DS_1, DS_2 filter Me_1 > 0);",
        datasets=["DS_1", "DS_2"],
        component_or_scalar=["Me_1"],
    ),
    # --- Validation operators ---
    "121": Classification(
        vtl='DS_r := check(DS_1#Me_1 > 0 errorcode "E001" errorlevel 1);',
        datasets=["DS_1"],
    ),
    "122": Classification(
        vtl='DS_r := check(not isnull(DS_1#Me_1) errorcode "E002" errorlevel 2 invalid);',
        datasets=["DS_1"],
    ),
    "123": Classification(
        vtl='DS_r := check(exists_in(DS_1, DS_2, true) errorcode "E003" errorlevel 3 all);',
        datasets=["DS_1", "DS_2"],
    ),
    "124": Classification(
        vtl=(
            "define datapoint ruleset dpr1 (variable Me_1 as Number) is\n"
            '  rule1: Me_1 > 0 errorcode "E001" errorlevel 1\n'
            "end datapoint ruleset;\n\n"
            "DS_r := check_datapoint(DS_1, dpr1);"
        ),
        datasets=["DS_1"],
    ),
    "125": Classification(
        vtl=(
            "define datapoint ruleset dpr1 (variable Me_1 as Number) is\n"
            '  rule1: Me_1 > 0 errorcode "E001" errorlevel 1\n'
            "end datapoint ruleset;\n\n"
            "DS_r := check_datapoint(DS_1, dpr1 all);"
        ),
        datasets=["DS_1"],
    ),
    "128": Classification(
        vtl='DS_r := check(DS_1 + SC_1 > 0 errorcode "E001" errorlevel 1);',
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    # --- Hierarchical operators ---
    "126": Classification(
        vtl=(
            "define hierarchical ruleset hr1 (variable rule Me_1) is\n"
            "  A = B + C\n"
            "end hierarchical ruleset;\n\n"
            "DS_r := hierarchy(DS_1, hr1 rule Me_1 non_null all);"
        ),
        datasets=["DS_1"],
    ),
    "127": Classification(
        vtl=(
            "define hierarchical ruleset hr1 (variable rule Me_1) is\n"
            "  A = B + C\n"
            "end hierarchical ruleset;\n\n"
            "DS_r := check_hierarchy(DS_1, hr1 rule Me_1 non_null all);"
        ),
        datasets=["DS_1"],
    ),
    # --- Dataset-only operators with mixed sub-expressions ---
    # When a dataset-only operator wraps a complex expression (e.g., DS_1 + SC_1),
    # the sub-expression operands should NOT all be forced to datasets.
    "129": Classification(
        vtl="DS_r := count(DS_1 + SC_1);",
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    "130": Classification(
        vtl="DS_r := sum(DS_1 * SC_1);",
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    "131": Classification(
        vtl="DS_r := avg(DS_1 + SC_1);",
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    "132": Classification(
        vtl="DS_r := flow_to_stock(DS_1 + SC_1);",
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    "133": Classification(
        vtl="DS_r := stock_to_flow(DS_1 - SC_1);",
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    "134": Classification(
        vtl="DS_r := union(DS_1 + SC_1, DS_2);",
        datasets=["DS_2"],
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    "135": Classification(
        vtl="DS_r := intersect(DS_1 + SC_1, DS_2);",
        datasets=["DS_2"],
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    "136": Classification(
        vtl="DS_r := union(DS_1 + SC_1, DS_2 + SC_2);",
        dataset_or_scalar=["DS_1", "DS_2", "SC_1", "SC_2"],
    ),
    "137": Classification(
        vtl="DS_r := exists_in(DS_1 + SC_1, DS_2, all);",
        datasets=["DS_2"],
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    "138": Classification(
        vtl="DS_r := timeshift(DS_1 + SC_1, 1);",
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    "139": Classification(
        vtl="DS_r := (DS_1 + SC_1)#Me_1;",
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    # Calc clause with dataset-only sub-expression: dataset is direct VarID,
    # but calc body contains sub-expression with external refs
    "140": Classification(
        vtl="DS_r <- DS_1[calc Me_2 := sum(Me_1 + SC_1)];",
        datasets=["DS_1"],
        component_or_scalar=["Me_1", "SC_1"],
    ),
    # --- Multi-statement ambiguity propagation ---
    # Intermediate from mixed expr fed to aggregation: ambiguity propagates
    "141": Classification(
        vtl="DS_A := DS_1 + SC_1;\nDS_r := sum(DS_A);",
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    # Intermediate from union fed to flow_to_stock with scalar
    "142": Classification(
        vtl="DS_A := union(DS_1, DS_2);\nDS_r := flow_to_stock(DS_A + SC_1);",
        datasets=["DS_1", "DS_2"],
        dataset_or_scalar=["SC_1"],
    ),
    # Intermediate from mixed expr fed to timeshift
    "143": Classification(
        vtl="DS_A := DS_1 + SC_1;\nDS_r := timeshift(DS_A, 1);",
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    # --- Nested dataset-only operators ---
    # count(union(...)): inner union marks datasets, count wraps it
    "144": Classification(
        vtl="DS_r := count(union(DS_1, DS_2));",
        datasets=["DS_1", "DS_2"],
    ),
    # union result combined with scalar in outer expression
    "145": Classification(
        vtl="DS_r := union(DS_1, DS_2) + SC_1;",
        datasets=["DS_1", "DS_2"],
        dataset_or_scalar=["SC_1"],
    ),
    # --- fill_time_series / period_indicator with expression ---
    "146": Classification(
        vtl="DS_r := fill_time_series(DS_1 + SC_1, all);",
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    "147": Classification(
        vtl="DS_r := period_indicator(DS_1 + SC_1);",
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    # --- setdiff / symdiff with expressions ---
    "148": Classification(
        vtl="DS_r := setdiff(DS_1 + SC_1, DS_2);",
        datasets=["DS_2"],
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
    "149": Classification(
        vtl="DS_r := symdiff(DS_1 + SC_1, DS_2 + SC_2);",
        dataset_or_scalar=["DS_1", "DS_2", "SC_1", "SC_2"],
    ),
    # --- check / validation with mixed sub-expressions ---
    # check wrapping aggregation: sum forces DS_1 to dataset
    "150": Classification(
        vtl='DS_r := check(sum(DS_1) > 0 errorcode "E001" errorlevel 1);',
        datasets=["DS_1"],
    ),
    # check with membership + scalar: DS_1 is dataset (membership), SC_1 ambiguous
    "151": Classification(
        vtl='DS_r := check(DS_1#Me_1 + SC_1 > 0 errorcode "E001" errorlevel 1);',
        datasets=["DS_1"],
        dataset_or_scalar=["SC_1"],
    ),
    # --- Clause operators with external scalar in expression ---
    # filter with external scalar ref
    "152": Classification(
        vtl="DS_r <- DS_1[filter Me_1 + SC_1 > 0];",
        datasets=["DS_1"],
        component_or_scalar=["Me_1", "SC_1"],
    ),
    # aggr with external scalar in aggregation body
    "153": Classification(
        vtl="DS_r := DS_1[aggr Me_1 := sum(Me_2 + SC_1) group by Id_1];",
        datasets=["DS_1"],
        component_or_scalar=["Me_2", "SC_1"],
    ),
    # --- Membership on expression result ---
    "154": Classification(
        vtl="DS_r := (DS_1 * SC_1)#Me_1;",
        dataset_or_scalar=["DS_1", "SC_1"],
    ),
}


_SORTED_CODES = sorted(CASES.keys(), key=lambda k: int(k))


@pytest.mark.parametrize("test_code", _SORTED_CODES)
def test_classification(test_code: str) -> None:
    case = CASES[test_code]
    if case.vtl is not None:
        script = case.vtl
    else:
        with open(data_path / "vtl" / f"{test_code}.vtl") as f:
            script = f.read()

    schedule = DAGAnalyzer.ds_structure(create_ast(script))

    assert sorted(schedule.global_input_datasets) == case.datasets
    assert sorted(schedule.global_input_scalars) == case.scalars
    assert sorted(schedule.global_input_dataset_or_scalar) == case.dataset_or_scalar
    assert sorted(schedule.global_input_component_or_scalar) == case.component_or_scalar
    # global_inputs is the union of all four categories
    all_classified = sorted(
        case.datasets + case.scalars + case.dataset_or_scalar + case.component_or_scalar
    )
    assert sorted(schedule.global_inputs) == all_classified
