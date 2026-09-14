"""V16 split. Extends the V11 split with OASIS-3.

The 84 AD-vs-HC cohort heads (dataset_OASIS/oasis3_ad_vs_hc/selected_sessions.csv) are held OUT as
the external clinical test set -- they must never enter training, or the AD/HC evaluation is void.
The remaining 125 OASIS heads go 100 train / 25 val, which is exactly what makes the totals come out
at 285 train / 45 val.

Head order is randomised by crc32("v16"+head) before splitting: OASIS-3 subject IDs correlate with
acquisition order and cohort, so an ID-ordered split would not be exchangeable
(memory: oasis3-cohort-composition).

Selection is FROZEN here as an explicit list -- regenerating it from a hash at import time would let
a change of Python or of the file glob silently move heads between splits mid-project.
"""
import v11_split as _V11

OASIS_PREFIX = "\0"          # disabled: OASIS heads are assigned explicitly below, not by prefix

OASIS_TEST  = ['oas30026', 'oas30043', 'oas30046', 'oas30057', 'oas30061', 'oas30075', 'oas30089', 'oas30098', 'oas30100', 'oas30119', 'oas30128', 'oas30139', 'oas30149', 'oas30173', 'oas30187', 'oas30194', 'oas30206', 'oas30208', 'oas30217', 'oas30296', 'oas30315', 'oas30316', 'oas30336', 'oas30342', 'oas30369', 'oas30387', 'oas30391', 'oas30394', 'oas30410', 'oas30411', 'oas30418', 'oas30423', 'oas30434', 'oas30438', 'oas30445', 'oas30449', 'oas30453', 'oas30479', 'oas30522', 'oas30534', 'oas30547', 'oas30556', 'oas30587', 'oas30665', 'oas30667', 'oas30680', 'oas30683', 'oas30731', 'oas30733', 'oas30768', 'oas30775', 'oas30796', 'oas30812', 'oas30823', 'oas30827', 'oas30841', 'oas30866', 'oas30884', 'oas30899', 'oas30926', 'oas30931', 'oas30948', 'oas30953', 'oas30959', 'oas30966', 'oas30970', 'oas30982', 'oas31003', 'oas31006', 'oas31020', 'oas31054', 'oas31086', 'oas31096', 'oas31104', 'oas31123', 'oas31125', 'oas31129', 'oas31139', 'oas31164', 'oas31175', 'oas31211', 'oas31301', 'oas31376', 'oas31398']

OASIS_TRAIN = ['oas30001', 'oas30003', 'oas30048', 'oas30080', 'oas30115', 'oas30117', 'oas30129', 'oas30137', 'oas30143', 'oas30178', 'oas30184', 'oas30219', 'oas30233', 'oas30240', 'oas30290', 'oas30304', 'oas30306', 'oas30350', 'oas30367', 'oas30402', 'oas30435', 'oas30443', 'oas30458', 'oas30462', 'oas30464', 'oas30483', 'oas30494', 'oas30502', 'oas30525', 'oas30548', 'oas30558', 'oas30572', 'oas30588', 'oas30589', 'oas30600', 'oas30612', 'oas30637', 'oas30659', 'oas30663', 'oas30673', 'oas30688', 'oas30699', 'oas30706', 'oas30721', 'oas30722', 'oas30725', 'oas30726', 'oas30735', 'oas30748', 'oas30751', 'oas30755', 'oas30764', 'oas30776', 'oas30788', 'oas30816', 'oas30821', 'oas30822', 'oas30839', 'oas30846', 'oas30862', 'oas30867', 'oas30900', 'oas30913', 'oas30921', 'oas30943', 'oas30960', 'oas30964', 'oas30978', 'oas30995', 'oas31010', 'oas31011', 'oas31014', 'oas31019', 'oas31022', 'oas31028', 'oas31041', 'oas31048', 'oas31071', 'oas31110', 'oas31136', 'oas31149', 'oas31158', 'oas31196', 'oas31214', 'oas31224', 'oas31225', 'oas31264', 'oas31267', 'oas31274', 'oas31293', 'oas31335', 'oas31336', 'oas31340', 'oas31354', 'oas31365', 'oas31382', 'oas31412', 'oas31435', 'oas31450', 'oas31459']

OASIS_VAL   = ['oas30065', 'oas30114', 'oas30134', 'oas30241', 'oas30249', 'oas30324', 'oas30361', 'oas30386', 'oas30392', 'oas30414', 'oas30508', 'oas30555', 'oas30571', 'oas30586', 'oas30618', 'oas30620', 'oas30765', 'oas30820', 'oas30845', 'oas30945', 'oas31103', 'oas31113', 'oas31195', 'oas31217', 'oas31442']

TEST_HEADS  = list(_V11.TEST_HEADS) + OASIS_TEST
VAL_HEADS   = list(_V11.VAL_HEADS)  + OASIS_VAL
TRAIN_HEADS = list(_V11.TRAIN_HEADS) + OASIS_TRAIN
HEADS       = TRAIN_HEADS + VAL_HEADS + TEST_HEADS


def source_of(h):
    # v11_split.source_of has no OASIS branch and silently returns "bw" for oas* -- which would
    # mis-weight 209 heads under FM_BALANCE=sqrt.
    if h.startswith("oas"):
        return "oas"
    return _V11.source_of(h)


def summary():
    from collections import Counter
    def bd(hs):
        c = Counter(source_of(h) for h in hs)
        return f"{len(hs)} (sh {c['sh']} scb {c['scb']} bw {c['bw']} oas {c['oas']})"
    return (f"TRAIN {bd(TRAIN_HEADS)}\nVAL   {bd(VAL_HEADS)}\nTEST  {bd(TEST_HEADS)}\n"
            f"total {len(HEADS)}  disjoint={len(set(HEADS))==len(HEADS)}")
