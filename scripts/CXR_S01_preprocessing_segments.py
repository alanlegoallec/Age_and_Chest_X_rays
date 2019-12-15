#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:00:27 2019

@author: Alan
"""

from CXR_helpers import *
from PIL import Image
from skimage.color import gray2rgb

#default args
if len(sys.argv)==1:
    sys.argv.append('2')
    sys.argv.append('299')
    
#run this script for each segment, i
i = int(sys.argv[1])
i = "{0:0=3}".format(i)
#resample to different images sizes
image_size = sys.argv[2]
  
#create a dictionnary for the download  
web_links = np.array([
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
	'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
	'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
	'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
	'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
	'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
	'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
	'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'])

dict_links = {}
for k in range(12):
    dict_links["{0:0=3}".format(k+1)] = web_links[k]

#download the file if necessary
if not os.path.isfile(path_compute + "images_" + i + ".tar.gz"):
    import urllib.request, urllib.parse, urllib.error
    url = 'http://www.blog.pythonlibrary.org/wp-content/uploads/2012/06/wxDbViewer.zip'
    print("downloading the data")
    urllib.request.urlretrieve(dict_links[i], path_compute + "images_" + i + ".tar.gz")

#extract the data
if not os.path.exists(path_compute + "images_" + i):
    print("extracting the data")
    os.mkdir(path_compute + "images_" + i)
    with tarfile.open(path_compute + "images_" + i + ".tar.gz") as tar:
        tar.extractall(path=path_compute + "images_" + i)

#list images       
dir_images = path_compute + "images_" + i + '/images/'
all_files = os.listdir(dir_images)

# read in the labels
labels = pd.read_csv(path_store + 'Data_Entry_2017.csv', header=0, index_col=0)
#only keep the labels of the segment of the data that is being analyzed
labels = labels.loc[all_files]
#get rid of unhealthy samples
labels = labels.loc[labels['Finding Labels'] == "No Finding",:]
#remove NA for age
labels = labels[labels['Patient Age'].notnull()]
#remove patients who are older than 122 years (some patients are 411 years old, entry error)
labels = labels[labels['Patient Age'] <= 122]

# delete useless files
print("delete useless files")
files = labels.index.values
for f in all_files:
  if f not in files:
    os.remove(dir_images + f)
#remove pictures which are dimensions (dim,dim,4) instead of (dim,dim)
to_remove_images = []
for f in files:
  img = Image.open(dir_images + f)
  if len(np.array(img).shape) != 2:
    to_remove_images.append(f)
    os.remove(dir_images + f)

#take the subset of the images and the labels accordingly
files = [e for e in files if e not in to_remove_images]
labels = labels.drop(to_remove_images)

# reorder the labels so that they line up with the order of the image files
labels = labels.reindex(files)

#resize the images
print("resize the images")
#Set parameter to resize the images
image_dim = (int(image_size),int(image_size)) #this number is matching the initial training of VVG16
#generate Xs in gray scale
images = []
for f in files:
    img = Image.open(dir_images + f)
    img = img.resize(image_dim)
    img_arr = np.array(img)
    images.append(img_arr)
Xs_grey = np.array(images)
#Convert the grey images to 3D rgb images, since VGG was trained on these.
Xs = gray2rgb(Xs_grey)

#Generate y
Ys = np.array(labels['Patient Age'])

#save files
print("save the files")
if not os.path.exists("data"):
    os.mkdir('data')
np.save(path_compute + 'X_' + image_size + '_' + i, Xs)
np.save(path_compute + 'y_' + i, Ys)
labels.to_pickle(path_compute + 'labels_' + i + '.pkl')
    
#clean the useless folders
#print("clean")
#if os.path.exists(path_compute + "images"):
#    shutil.rmtree(path_compute + "images")

print("done")

