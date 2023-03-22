import pandas as pd
import os

sampler_seed = 11123 # approx equal % of data sampled across all 7 classes
test_dir = "/home/lixin/Classes/Spr23/542/Projects-ECE542/Group/TestData/test/"
test_dir_source = "/home/lixin/Classes/Spr23/542/Projects-ECE542/Group/TestData/original/"
filenames = os.listdir(test_dir_source)
labels = pd.read_csv('/home/lixin/Classes/Spr23/542/Projects-ECE542/Group/TestData/ISIC2018_Task3_Test_GroundTruth.csv')


#print(labels.head())
X_y = labels.get(['image_id','dx']) # image_id points to input image filename, dx is the class output
#print(X_y.head())
#print('\nData Distribution:\n',X_y['dx'].value_counts()) # show data imbalance, 60:1 with nv and df classes
                                # ignore for now

#train_set = pd.DataFrame.sample(X_y, train_set_count, replace=False, random_state=sampler_seed)
#print(train_set['dx'])
#val_set = pd.DataFrame.drop_duplicates(pd.concat([X_y, train_set]), keep=False)
#print(val_set['dx'])
#print(val_set['dx'].value_counts()/train_set['dx'].value_counts())

for i, row in labels.iterrows():
    image = row['image_id']
    dx = row['dx']
    filename = image +'.jpg'
    os.popen('mkdir -p '+test_dir+dx)
    if filename in filenames:
        cmd = 'cp '+test_dir_source+filename+' '+test_dir+dx+'/'
        print(f"{i}: {cmd}")
        os.popen(cmd)
    else:
        print(f"File {filename} not found")
        
