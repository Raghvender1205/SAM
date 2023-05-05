import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def show_annotations(annotations):
    if len(annotations) == 0:
        return 
    sorted_annotations = sorted(annotations, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for i in sorted_annotations:
        m = i['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for j in range(3):
            img[:, :, j] = color_mask[j]
        out = np.dstack((img, m*0.35))
        ax.imshow(out)

if __name__ == '__main__':
    # sam = sam_model_registry['vit_h'](checkpoint="weights/sam_vit_h_4b8939.pth")
    sam = sam_model_registry['vit_b'](checkpoint='weights/sam_vit_b_01ec64.pth')
    sam.cuda()
    mask_gen = SamAutomaticMaskGenerator(sam)

    image = cv2.imread('public.jpg')
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_gen.generate(img)
    plt.figure(figsize=(12, 9))
    plt.imshow(img)
    show_annotations(masks)
    plt.axis('off')
    plt.savefig(os.path.join('outputs', 'result.jpg'), bbox_inches='tight')