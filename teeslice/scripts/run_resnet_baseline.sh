


for ARCH in resnet18
do
for MODE in Enclave CPU GPU
do

python -m teeslice.sgx_resnet_cifar \
--arch $ARCH \
--mode $MODE \
--batch_size 64 \

done
done