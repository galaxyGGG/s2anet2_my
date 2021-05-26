from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = '/workspace/s2anet/configs/dota/s2anet_r50_fpn_1x_dota.py'
checkpoint_file = '/data/checkpoints/s2anet_r50_fpn_1x_converted-11c9c5f4.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/workspace/s2anet/demo/demo.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
show_result(img, result, model.CLASSES)
# or save the visualization results to image files
# show_result(img, result, model.CLASSES, out_file='result.jpg')