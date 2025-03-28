import os

data_dir = "../Inspaint_data/random_medium_256"
for img in os.listdir(data_dir):
    if img == "Masked" or img == "Target":
        continue
    if img.__contains__("mask"):
        os.rename(os.path.join(data_dir, img), os.path.join(data_dir,"Masked", img))
    else:
        os.rename(os.path.join(data_dir, img), os.path.join(data_dir,"Target", img))

target_path = "../Inspaint_data/random_medium_256/Target"
masked_path = "../Inspaint_data/random_medium_256/Masked"

### Renamed the files to the same format as existing data
# for filename in os.listdir(target_path):
#     new_filename = filename.split("_")[0]+".png"
#     os.rename(os.path.join(target_path, filename), os.path.join(target_path, new_filename))

# for filename in os.listdir(masked_path):
#     new_filename = filename.replace("target", "mask").split("_")[0]+".png"
#     os.rename(os.path.join(masked_path, filename), os.path.join(masked_path, new_filename))

parent_dir = "../Inspaint_data"
### Move the files to Train_Data folder
for filename in os.listdir(target_path):
    os.rename(os.path.join(target_path, filename), os.path.join(parent_dir,"Train_Data/Target", filename))
for filename in os.listdir(masked_path):
    os.rename(os.path.join(masked_path, filename), os.path.join(parent_dir,"Train_Data/Masked", filename))