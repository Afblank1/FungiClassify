import os

data_directory = 'FungiClassify\extracted_defungi'
class_counts = {}

# Loop through each class folder
for class_name in os.listdir(data_directory):
    class_folder = os.path.join(data_directory, class_name)
    if os.path.isdir(class_folder):
        # Count the number of image files in the class folder
        count = len([
            f for f in os.listdir(class_folder)
            if os.path.isfile(os.path.join(class_folder, f))
        ])
        class_counts[class_name] = count

# Print the class counts
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")
