_base_ = ['./maptrv2_flex_nusc_r50_24ep.py']

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            encoder=dict(
                tie_layer_weights=True,
                num_train_iters=4,
                num_test_iters=4,
            ))))
