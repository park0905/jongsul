import os
imdir = 'images'
if not os.path.isdir(imdir):
    os.mkdir(imdir)

fight_folder = [folder for folder in os.listdir('.') if 'dong' in folder]


n = 0
for folder in fight_folder:
    for imfile in os.scandir(folder):
        os.rename(imfile.path, os.path.join(imdir, '{:06}.png'.format(n)))
        n += 1
