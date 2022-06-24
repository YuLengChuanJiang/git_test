import numpy as np
rgb_score = np.load('/home/ange/projects/tsn-ms-pytorch/tsn-pytorch20200722/resnet_rgb_score_seg5_0720.npz', encoding='bytes')

labels = rgb_score['labels']
print(labels)
scores = rgb_score['scores']
pred_label = []
for x in scores:
    video_pred = np.argmax(np.mean(x[0], axis=0))
    pred_label.append(video_pred)
#video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in scores]
scores[0,0]
print(scores)
