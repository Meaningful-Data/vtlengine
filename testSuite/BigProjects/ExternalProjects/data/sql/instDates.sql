SELECT CNTRCT_ID, CRDTR_ID, DBTR_ID, DT_RFRNC, INSTRMNT_ID, OBSRVD_AGNT_CD,
         julianday(DT_END_INTRST_ONLY) - julianday(DT_RFRNC) as NXT_INTRST_RT_RST_DYS,
         julianday(DT_LGL_FNL_MTRTY) - julianday(DT_INCPTN) as ORGNL_MTRTY, 
         julianday(DT_LGL_FNL_MTRTY) - julianday(DT_RFRNC) as RSDL_MTRTY, 
         julianday(DT_RFRNC) - julianday(DT_PST_D) as PST_D_DYS
   
FROM ANAMART_INSTRMNT_DYS_P;