python3 train.py --n_epochs 10 \
                  --root /root/M11115Q13/dataset/image_size1024/ \
                  --batch_size 32 \
                  --model convnext_base \
                  --optimizer sgd \
                  --lr 7e-4 \
                  --load /root/M11115Q13/crop_classification/checkpoints/convnext_base/model_epoch19_acc0.8814.pth

python3 train.py --n_epochs 10 \
                  --root /root/M11115Q13/dataset/image_size1024/ \
                  --batch_size 8 \
                  --model convnext_small \
                  --optimizer sgd \
                  --lr 7e-4 \
                  --load /root/M11115Q13/crop_classification/checkpoints/convnext_small/model_epoch19_acc0.8791.pth

python3 train.py --n_epochs 10 \
                  --root /root/M11115Q13/dataset/image_size1024/ \
                  --batch_size 16 \
                  --model resnext50 \
                  --lr 7e-4 \
                  --device 2 \
                  --load /root/M11115Q13/crop_classification/checkpoints/resnext50/model_epoch19_acc0.8603.pth

