mkdir dataset
# Download train, validation, test datasets using wget
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip?download=true -O dataset/train.zip -c
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip?download=true -O dataset/val.zip -c
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_test.zip?download=true -O dataset/test.zip -c
#widerfice annotation
wget http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip -O dataset/wider_face_split.zip -c

gdown 1tU_IjyOwGQfGNUvZGwWWM4SwxKp2PUQ8 -O dataset/retinaface.zip

unzip -oq dataset/test.zip -d dataset
unzip -oq dataset/train.zip -d dataset
unzip -oq dataset/val.zip -d dataset
mv dataset/WIDER_test dataset/test
mv dataset/WIDER_val dataset/val
mv dataset/WIDER_train dataset/train
unzip -oq dataset/wider_face_split.zip -d dataset/anno
unzip -oq dataset/retinaface.zip -d dataset

cd data
mkdir widerface
mkdir widerface/val
mkdir widerface/train

python train2yolo.py ../dataset/train ./widerface/train
python val2yolo.py  ../dataset/  ./widerface/val