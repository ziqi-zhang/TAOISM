


for ARCH in resnet18
do

python -m teeslice.eval_sgx_teeslice \
--arch $ARCH \
--batch_size 64 \
--root teeslice/cifar10val \
--pretrained_dataset cifar100 \


done
