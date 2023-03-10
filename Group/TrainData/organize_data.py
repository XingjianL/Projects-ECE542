import pandas as pd
import os

sampler_seed = 11123 # approx equal % of data sampled across all 7 classes
train_dir = "/home/lixin/Classes/Spr23/542/Projects-ECE542/Group/TrainData/train/"
val_dir = "/home/lixin/Classes/Spr23/542/Projects-ECE542/Group/TrainData/validation/"

part1_dir = "/home/lixin/Classes/Spr23/542/Projects-ECE542/Group/TrainData/archive/HAM10000_images_part_1/"
part1_filename = os.listdir(part1_dir)

part2_dir = "/home/lixin/Classes/Spr23/542/Projects-ECE542/Group/TrainData/archive/HAM10000_images_part_2/"
part2_filename = os.listdir(part2_dir)
labels = pd.read_csv('/home/lixin/Classes/Spr23/542/Projects-ECE542/Group/TrainData/archive/HAM10000_metadata.csv')

total_files = len(part1_filename) + len(part2_filename)
print(f"Total number of data: {total_files}")
test_set_count = 1511
val_set_count = test_set_count
train_set_count = total_files - val_set_count
print(f"Proportion: test:{test_set_count/(total_files+test_set_count):.3f} train:{train_set_count/(total_files+test_set_count):.3f} val:{val_set_count/(total_files+test_set_count):.3f}")

#print(labels.head())
X_y = labels.get(['image_id','dx']) # image_id points to input image filename, dx is the class output
#print(X_y.head())
#print('\nData Distribution:\n',X_y['dx'].value_counts()) # show data imbalance, 60:1 with nv and df classes
                                # ignore for now

train_set = pd.DataFrame.sample(X_y, train_set_count, replace=False, random_state=sampler_seed)
#print(train_set['dx'])
val_set = pd.DataFrame.drop_duplicates(pd.concat([X_y, train_set]), keep=False)
#print(val_set['dx'])
print(val_set['dx'].value_counts()/train_set['dx'].value_counts())

for i, row in train_set.iterrows():
    image = row['image_id']
    dx = row['dx']
    filename = image +'.jpg'
    os.popen('mkdir -p '+train_dir+dx)
    if filename in part1_filename:
        cmd = 'cp '+part1_dir+filename+' '+train_dir+dx+'/'
        print(f"{i}: {cmd}")
        os.popen(cmd)
    elif filename in part2_filename:
        cmd = 'cp '+part2_dir+filename+' '+train_dir+dx+'/'
        print(f"{i}: {cmd}")
        os.popen(cmd)
    else:
        print(f"File {filename} not found")
        break
for i, row in val_set.iterrows():
    image = row['image_id']
    dx = row['dx']
    filename = image +'.jpg'
    os.popen('mkdir -p '+val_dir+dx)
    if filename in part1_filename:
        cmd = 'cp '+part1_dir+filename+' '+val_dir+dx+'/'
        print(f"{i}: {cmd}")
        os.popen(cmd)
    elif filename in part2_filename:
        cmd = 'cp '+part2_dir+filename+' '+val_dir+dx+'/'
        print(f"{i}: {cmd}")
        os.popen(cmd)
    else:
        print(f"File {filename} not found")
        break