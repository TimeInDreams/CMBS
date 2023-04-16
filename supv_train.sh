python supv_main.py \
--gpu 0 \
--lr 0.0007 \
--clip_gradient 0.5 \
--snapshot_pref "../drive/MyDrive/Exps/Supv/exp1/" \
--resume "../drive/MyDrive/Exps/Supv/exp1/model_epoch_11_top1_75.672_task_Supervised_best_model.pth.tar"
--n_epoch 200 \
--b 128 \
--test_batch_size 16 \
--print_freq 1
