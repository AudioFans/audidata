# This file is modified from https://github.com/MorenoLaQuatra/ARCH/blob/main/data_download/download_data.sh

ROOT="/datasets" # replace with the directory where you want to store the data
ROOT="/home/jiahelei/DATASETS/TEST"
mkdir -p ${ROOT}

cd ${ROOT}

# -------------------------- MagnaTagATune --------------------------
echo "Downloading MagnaTagATune dataset..."

mkdir -p magnatagatune
cd magnatagatune

# 1. Download dataset files
wget -c https://mirg.city.ac.uk/datasets/magnatagatune/clip_info_final.csv
wget -c https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.001
wget -c https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.002
wget -c https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.003
wget -c https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv

# 2. Combine and unzip the mp3 files
cat mp3.zip.* > mp3.zip
unzip mp3.zip
rm mp3.zip
rm mp3.zip.*

# 3.Download the split files
wget -c https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/test_gt_mtt.tsv
wget -c https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/train_gt_mtt.tsv
wget -c https://raw.githubusercontent.com/jordipons/musicnn-training/master/data/index/mtt/val_gt_mtt.tsv

echo "MagnaTagATune dataset downloaded successfully."

cd ${ROOT}

# -------------------------- Medley-Solos-DB --------------------------
echo "Downloading MedleyDB dataset..."

mkdir -p medleydb
cd medleydb

wget -c https://zenodo.org/record/3464194/files/Medley-solos-DB.tar.gz
wget -c https://zenodo.org/record/2582103/files/Medley-solos-DB_metadata.csv

mkdir audio
mv Medley-solos-DB.tar.gz audio/
cd audio
tar -xvzf Medley-solos-DB.tar.gz
rm Medley-solos-DB.tar.gz

echo "MedleyDB dataset downloaded successfully."
cd ${ROOT}
