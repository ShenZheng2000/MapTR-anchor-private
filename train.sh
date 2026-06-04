# export PYTHONPATH=/home/shenzheng_google_com/Projects/Inf_Perception/Methods/MapTR:$PYTHONPATH

# => train maptrv2 on original split
# bash ./tools/dist_train.sh \
#     ./projects/configs/maptrv2/maptrv2_av2_3d_r50_6ep.py \
#     8 \
#     --work-dir work_dirs/maptrv2_av2_3d_r50_6ep

# # # => train maptrv2 on geosplit
# bash ./tools/dist_train.sh \
#     ./projects/configs/maptrv2/maptrv2_av2_3d_r50_6ep_geosplit.py \
#     8 \
#     --work-dir work_dirs/maptrv2_av2_3d_r50_6ep_geosplit

# # # # => train maptrv2 on map change
# bash ./tools/dist_train.sh \
#     ./projects/configs/maptrv2/maptrv2_mapchange_3d_r50_6ep.py \
#     8 \
#     --work-dir work_dirs/maptrv2_mapchange_3d_r50_6ep

# # # => train maptrv2 on geosplit (nuscenes)
# bash ./tools/dist_train.sh \
#     ./projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
#     8 \
#     --work-dir work_dirs/maptrv2_nusc_r50_24ep

# => train maptrv2 + FlexSceneEncoder (nuscenes)
# NOTE: use 2 gpu for quick debug! 
bash ./tools/dist_train.sh \
    ./projects/configs/maptrv2/maptrv2_flex_nusc_r50_24ep.py \
    8 \
    --work-dir work_dirs/maptrv2_flex_nusc_r50_24ep