ENV='in'
CR=8

python main.py \
--evaluate \
--pretrained `printf "../../csi_reference/CRNet/history/CRnet_checkpoint_%s_%02d.model" ${ENV} ${CR}` \
--scenario ${ENV} \
--batch_size 200 \
--workers 0 \
--cr ${CR} \
--scheduler cosine \
--gpu 0