import os
from dataset import create_generators

d = 'dataset_images'

print('--- class counts ---')
for c in sorted(os.listdir(d)):
    p = os.path.join(d, c)
    if os.path.isdir(p):
        cnt = sum(1 for f in os.listdir(p) if f.lower().endswith(('.jpg','.png','.jpeg')))
        print(f"{c}: {cnt}")

print('\n--- generator sample ---')
tg, vg = create_generators(d, img_size=(224,224), batch_size=8)
print('class_indices:', tg.class_indices)
xb, yb = next(tg)
print('x shape:', xb.shape)
print('y shape:', yb.shape)
print('example y[0]:', yb[0])