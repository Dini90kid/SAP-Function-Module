
from lakehouse_fm_agent.core import alpha, uom, currency
from lakehouse_fm_agent.bw_replace import dso, logsys

REGISTRY = {
    'CONVERSION_EXIT_ALPHA_INPUT': alpha.alpha_input,
    'UNIT_CONVERSION_SIMPLE': uom.convert_simple,
    'CONVERT_TO_LOCAL_CURRENCY': currency.convert_fx,
    'RSDRI_ODSO_UPDATE': dso.merge_into_delta,
    'RSDG_LOGSYS_GET_FROM_ID': logsys.map_logsys_from_id,
    'Y_DNP_CONV_BUOM_SU_SSU': uom.conv_buom_su_ssu,
    'YDNP_CHK_UOM_1': uom.check_uom,
}

POINTERS = {
    'RRSI_VAL_SID_SINGLE_CONVERT': 'No SIDs in Lakehouse; join on business keys to dimension tables.',
    'DDIF_FIELDINFO_GET': 'Use Spark schema & control metadata tables.',
    'RSDMD_WRITE_ATTRIBUTES_TEXTS': 'Maintain attributes/texts in Delta; MERGE for upserts.'
}

def resolve_handler(fm:str):
    return REGISTRY.get(fm.upper())
