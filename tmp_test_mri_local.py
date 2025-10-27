from PIL import Image
import numpy as np
import cv2


def is_probable_mri_local(pil_image, target_size=(224, 224)):
    try:
        img = pil_image.convert('RGB').resize(target_size)
        arr = np.array(img)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        mean_sat = float(np.mean(hsv[..., 1])) / 255.0
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        pad = max(8, min(h, w) // 16)
        corners = [gray[0:pad, 0:pad], gray[0:pad, w - pad:w], gray[h - pad:h, 0:pad], gray[h - pad:h, w - pad:w]]
        corner_darkness = np.mean([np.mean(c < 30) for c in corners])
        cx0, cx1 = h // 4, 3 * h // 4
        cy0, cy1 = w // 4, 3 * w // 4
        center = gray[cx0:cx1, cy0:cy1]
        center_mean = float(np.mean(center)) / 255.0
        score = (1.0 - mean_sat) * 0.55 + corner_darkness * 0.30 + (1.0 - abs(center_mean - 0.5) * 2) * 0.15
        score = max(0.0, min(1.0, score))
        return score
    except Exception:
        return 0.0


# Test on a known MRI
mri_path = 'dataset_images/yes/y0.jpg'
img = Image.open(mri_path)
score_mri = is_probable_mri_local(img)
print('MRI score for', mri_path, '->', score_mri)

# Test on an example non-MRI: a random RGB photo
rand = (np.random.rand(224,224,3) * 255).astype('uint8')
img_rand = Image.fromarray(rand)
score_rand = is_probable_mri_local(img_rand)
print('Random image score ->', score_rand)
