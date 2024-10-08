SELECT CNTRCT_ID, DT_RFRNC, INSTRMNT_ID, PRTCTN_ID, OBSRVD_AGNT_CD,
              IS_PRTCTN_MTRTY_MSMTCH,
              CASE IS_PRTCTN_MTRTY_MSMTCH
                WHEN "T"
                  THEN julianday(DT_LGL_FNL_MTRTY) - julianday(DT_MTRTY_PRTCTN)
              END MTRTY_MSMTCH,
              julianday(DT_LGL_FNL_MTRTY) - julianday(DT_MTRTY_PRTCTN) as PRTCTN_RSDL_MTRTY_DYS
FROM MSMTCH_BL_DS;