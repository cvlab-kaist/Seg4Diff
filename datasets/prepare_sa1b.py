import torch
import os
import tqdm

sam_path='images/'
ann_path='annotations/'
save_filename='image_list.da'
folders = os.listdir(sam_path)

for folder in tqdm.tqdm(folders):
    curr_path = os.path.join(sam_path, folder)
    curr_ann = os.path.join(ann_path, folder)
    files = os.listdir(curr_path)
    save_path = os.path.join(curr_path, save_filename)
    f_save = open(save_path, 'wb')
    a = []
    for f in files:
        if f.split('.')[-1]=='jpg':
            a.append({'img_name': os.path.join(curr_path, f), 'ann_name': os.path.join(curr_ann, f.split('.')[0]+'.json'),
                      'file_name': os.path.join(curr_path, f)})
    torch.save(a, f_save)
    f_save.close()