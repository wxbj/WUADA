#########################full supervise##############################
# train：CT->MR
python train_full_supervise.py -cfg configs/full_supervise_ct2mr_u3plus_r101_WUADA.yaml
# test
python test.py -cfg configs/full_supervise_ct2mr_u3plus_r101_WUADA.yaml

# train：MR->CT
python train_full_supervise.py -cfg configs/full_supervise_mr2ct_u3plus_r101_WUADA.yaml
# test
python test.py -cfg configs/full_supervise_mr2ct_u3plus_r101_WUADA.yaml

# train: bSSFP->LGE
python train_full_supervise.py -cfg configs/full_supervise_bSSFP2LGE_u3plus_r101_WUADA.yaml
# test
python test.py -cfg configs/full_supervise_bSSFP2LGE_u3plus_r101_WUADA.yaml
############################WUADA:RA######################################
# train: CT-> MR
python train.py -cfg configs/RA_ct2mr_u3plus_r101_WUADA.yaml
# test
python test.py -cfg configs/RA_ct2mr_u3plus_r101_WUADA.yaml

# train: MR->CT
python train.py -cfg configs/RA_mr2ct_u3plus_r101_WUADA.yaml
# test
python test.py -cfg configs/RA_mr2ct_u3plus_r101_WUADA.yaml

# train: bSSFP->LGE
python train.py -cfg configs/RA_bssfp2lge_u3plus_r101_WUADA.yaml
# test
python test.py -cfg configs/RA_bssfp2lge_u3plus_r101_WUADA.yaml
#############################################WUADA:PA#################################
# train: CT-> MR
python train.py -cfg configs/PA_ct2mr_u3plus_r101_WUADA.yaml
# test
python test.py -cfg configs/PA_ct2mr_u3plus_r101_WUADA.yaml

# train: MR->CT
python train.py -cfg configs/PA_mr2ct_u3plus_r101_WUADA.yaml
# test
python test.py -cfg configs/PA_mr2ct_u3plus_r101_WUADA.yaml

# train: bSSFP->LGE
python train.py -cfg configs/PA_bssfp2lge_u3plus_r101_WUADA.yaml
# test
python test.py -cfg configs/PA_bssfp2lge_u3plus_r101_WUADA.yaml
########################################source only##################################
# train: CT->MR
python train_source_only.py -cfg configs/source_only_ct2mr_u3plus_r101_WUADA.yaml
# test
python test.py -cfg configs/source_only_ct2mr_u3plus_r101_WUADA.yaml

# train: MR->CT
python train_source_only.py -cfg configs/source_only_mr2ct_u3plus_r101_WUADA.yaml
# test
python test.py -cfg configs/source_only_mr2ct_u3plus_r101_WUADA.yaml

# train: bSSFP->LGE
python train_source_only.py -cfg configs/source_only_bssfp2lge_u3plus_r101_WUADA.yaml
# test
python test.py -cfg configs/source_only_bssfp2lge_u3plus_r101_WUADA.yaml

########################################source_free##########################################
# train RA:ct->mr
python train_source_free.py -cfg configs/source_free_RA_ct2mr_u3plus_r101_WUADA.yaml
# test
python test.py -cfg configs/source_free_RA_ct2mr_u3plus_r101_WUADA.yaml

# train PA:ct->mr
python train_source_free.py -cfg configs/source_free_PA_ct2mr_u3plus_r101_WUADA.yaml
# test
python test.py -cfg configs/source_free_PA_ct2mr_u3plus_r101_WUADA.yaml