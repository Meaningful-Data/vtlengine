import json
import os
from pathlib import Path

from testSuite.Helper import TestHelper


class BIRDHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class SemanticBIRD(BIRDHelper):
    """

    """

    classTest = 'BIRD_BIRD.SemanticBIRD'

    def test_D_ENTRPRS_SZ_CLCLTD_1(self):
        '''

        '''
        code = 'D_ENTRPRS_SZ_CLCLTD_1'
        number_inputs = 6
        references_names = []

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_INPUT_LAYER_TO_ENRICHED_INPUT_LAYER(self):
        '''

        '''
        code = 'INPUT_LAYER_TO_ENRICHED_INPUT_LAYER'
        number_inputs = 62
        # references_names = []
        references_names = ['ADVNCS_NT_LNS_E', 'INSTRMNTS_BNFCRS_E', 'INSTRMNTS_CRDTRS_E', 'INSTRMNTS_CSTMRS_E',
                            'INSTRMNTS_ORGNTRS_E', 'INSTRMNTS_PRTCTNS_E', 'INSTRMNTS_SRVCRS_E', 'INVSTMNT_PRPRTY_E',
                            'JNT_CNTRPRTS_E_', 'LNS_E', 'MSTR_NTTNG_AGRMNT_E', 'NN_FNNCL_ASSTS_E', 'NN_FNNCL_LBLTS_E',
                            'OTHR_ASSTS_E', 'OTHR_FNNCL_LBLTS_NT_FNNCL_GRNTS_LN_CMMTMNTS_GVN_E', 'OTHR_INTNGBL_ASSTS_E',
                            'OWND_SCRTS_E', 'P_ADVNCS_NT_LNS_E_1.ADVNCS_NT_LNS_E',
                            'P_CMMTMNTS_GVN_E_1.CMMTMNTS_GVN_UNN',
                            'P_CNTRPRTS_E_1.CNTRPRTS_E_JN', 'P_DPSTS_LBLTS_E_1.DPSTS_LBLTS_E', 'P_DRVTV_E_1.DRVTV_E',
                            'P_EQTY_SCRTS_E_1.EQTY_SCRTS_E', 'P_IMPLCT_CRDT_CRD_DBT_1.CRDT_CRD_DBT_E',
                            'P_IMPLCT_CRDT_DRVTV_1.CRDT_DRVTV_E', 'P_IMPLCT_CRDT_FCLTS_GVN_1.CRDT_FCLTS_E',
                            'P_IMPLCT_CRRNT_ACCNT_ASSTS_1.CRRNT_ACCNT_ASSTS_E',
                            'P_IMPLCT_CRRNT_ACCNT_LBLTS_1.CRRNT_ACCNT_LBLTS_E',
                            'P_IMPLCT_FCTRNG_1.FCTRNG_E', 'P_IMPLCT_FNNCL_GRNTS_GVN_1.FNNCL_GRNTS_GVN_E',
                            'P_IMPLCT_FNNCL_LSS_1.FNNCL_LSS_E',
                            'P_IMPLCT_FNNCL_OPTN_1.FNNCL_OPTN_E', 'P_IMPLCT_OTHR_CMMTMNTS_GVN_1.OTHR_CMMTMNTS_GVN_E',
                            'P_IMPLCT_OTHR_DPSTS_1.OTHR_DPSTS_E', 'P_IMPLCT_OTHR_FNNCL_DRVTV_1.OTHR_FNNCL_DRVTV_E',
                            'P_IMPLCT_OTHR_LNS_1.OTHR_LNS_E', 'P_IMPLCT_OTHR_TRD_RCVBLS_1.OTHR_TRD_RCVBLS_E',
                            'P_IMPLCT_RPRCHS_AGRMNT_1.RPRCHS_AGRMNT_E', 'P_IMPLCT_RVRS_RPRCHS_LNS_1.RVRS_RPRCHS_LNS_E',
                            'P_INSTRMNTS_SRVCRS_E_1.INSTRMNTS_SRVCRS_ADDTNL', 'P_LNS_E_1.LNS_E_FNL',
                            'P_LNS_E_1.LNS_E_PRMTR',
                            'P_LNS_E_1.LNS_E_UNN', 'P_OTHR_FNNCL_PRTCTN_E_1.OTHR_FNNCL_PRTCTN_E',
                            'P_OTHR_PHYSCL_PRTCTN_E_1.OTHR_PHYSCL_PRTCTN_E',
                            'P_OWND_SCRTS_E_1.OWND_SCRTS_E_P_CRRYNG_AMNT', 'P_OWND_SCRTS_E_1.OWND_SCRTS_JN',
                            'P_PL_ITM_E_1.PL_ITM_E',
                            'PRMTR.CNSTNT_IS_CRRYNG_AMNT_DRVD', 'PRPRTY_PLNT_EQPMNT_E', 'PRTCTNS_NT_RL_ESTT_E',
                            'PRTCTNS_PRTCTN_PRVDRS_E',
                            'PRVSN_FLWS_NT_FNNCL_GRNTS_CMMTMNTS_GVN_E', 'P_SCRTS_ISSD_E_1.SCRTS_ISSD_E',
                            'P_SCRTS_PRTCTN_E_1.SCRTS_PRTCTN_E',
                            'P_SCRTS_PRTCTN_E_1.SCRTS_PRTCTN_JN', 'RPRCHS_AGRMNT_LNS_E', 'RPRCHS_AGRMNT_SCRTS_E',
                            'SCRTSTNS_ORGNTRS_E',
                            'SCRTSTNS_OTHR_CRDT_TRNSFRS_E', 'SCRTSTNS_SRVCRS_E', 'SHRT_PSTNS_E',
                            'TRNSCTN_CNTRPRTS_SPLT.CMMTMNTS_RCVD_CSTMRS_RLTNSHP',
                            'TRNSCTN_CNTRPRTS_SPLT.CMMTMNTS_RCVD_CSTMRS_RLTNSHP_FNNCL_GRTS',
                            'TRNSCTN_CNTRPRTS_SPLT.INSTRMNTS_SRVCRS',
                            'TRNSCTN_CNTRPRTS_SPLT.PRTCTNS_PRTCTN_PRVDRS_E_1',
                            'TRNSCTN_CNTRPRTS_SPLT.PRTCTNS_PRTCTN_PRVDRS_E_2',
                            'CMMTMNT_RCVD_E', 'CMMTMNTS_GVN_CSTMRS_RLTNSHP_E', 'CMMTMNTS_GVN_E',
                            'CMMTMNTS_GVN_PRTCTNS_E',
                            'CMMTMNTS_RCVD_CSTMRS_RLTNSHP_E', 'CNTRPRTS_E', 'CNTRY_E', 'CRDT_FCLTS_E',
                            'CRDT_FCLTS_INSTRMNTS_E',
                            'CRRNT_ACCNT_SPLT.CRRNT_ACCNT_ASSTS', 'CRRNT_ACCNT_SPLT.CRRNT_ACCNT_LBLTS', 'CSH_ON_HND_E',
                            'DBT_SCRTS_E', 'D_CRRYNG_AMNT_SCRTS_1.D_CRRYNG_AMNT_SCRTS',
                            'D_CRRYNG_AMNT_SCRTS_1.OWND_SCRTS_PRMTR',
                            'D_ENTRPRS_SZ_CLCLTD_1.ALL0', 'D_ENTRPRS_SZ_CLCLTD_1.ALL1', 'D_ENTRPRS_SZ_CLCLTD_1.ALL2',
                            'D_ENTRPRS_SZ_CLCLTD_1.CNTRPRTS_ATNMS', 'D_ENTRPRS_SZ_CLCLTD_1.CNTRPRTS_ATNMS0',
                            'D_ENTRPRS_SZ_CLCLTD_1.CNTRPRTS_INPT0', 'D_ENTRPRS_SZ_CLCLTD_1.CNTRPRTS_INPT1',
                            'D_ENTRPRS_SZ_CLCLTD_1.CNTRPRTS_OBJCTV', 'D_ENTRPRS_SZ_CLCLTD_1.CNTRPRTS_PRVS',
                            'D_ENTRPRS_SZ_CLCLTD_1.D_ENTRPRS_SZ_CLCLTD', 'D_ENTRPRS_SZ_CLCLTD_1.D_ENTRPRS_SZ_CLCLTD0',
                            'D_ENTRPRS_SZ_CLCLTD_1.D_ENTRPRS_SZ_CLCLTD1', 'D_ENTRPRS_SZ_CLCLTD_1.FNL_STTNG',
                            'D_ENTRPRS_SZ_CLCLTD_1.FNL_STTNG2', 'D_ENTRPRS_SZ_CLCLTD_1.FNL_STTNG3',
                            'D_ENTRPRS_SZ_CLCLTD_1.GRP_ENTRPRSS_AGG',
                            'D_ENTRPRS_SZ_CLCLTD_1.GRP_ENTRPRSS_AGG0', 'D_ENTRPRS_SZ_CLCLTD_1.GRP_ENTRPRSS_AGG1',
                            'D_ENTRPRS_SZ_CLCLTD_1.PRLMNRY_STTNG', 'D_ENTRPRS_SZ_CLCLTD_1.PRTNR_ENTRPRSS_AGG',
                            'D_ENTRPRS_SZ_CLCLTD_1.TTL_AGG',
                            'D_ENTRPRS_SZ_CLCLTD_1.TTL_AGG0', 'D_ENTRPRS_SZ_CLCLTD_1.TTL_AGG1',
                            'D_ENTRPRS_SZ_CLCLTD_1.TTL_AGG2',
                            'D_ENTRPRS_SZ_CLCLTD_1.TTL_AGG3', 'D_ENTRPRS_SZ_CLCLTD_1.TTL_AGG4',
                            'D_ENTRPRS_SZ_CLCLTD_1.TTL_AGG5',
                            'D_ENTRPRS_SZ_CLCLTD_1.TTL_AGG6', 'DPSTS_THT_LBLTS_E', 'DRVTVS_CSTMRS_RLTSHP_E',
                            'EQTY_INSTRMNTS_NT_SCRTS_E',
                            'FCTRNG_AXLRY_E', 'GDWLL_E', 'GRP_E']

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
