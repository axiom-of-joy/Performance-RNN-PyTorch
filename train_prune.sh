python3 train_prune.py \
-s save/LSTM_model.sess \
-d dataset/processed/ecomp_piano \
-i 10 \
-C distiller/examples/agp-pruning/word_lang_model.LARGE_70.schedule_agp.yaml \
#-D sparsity
