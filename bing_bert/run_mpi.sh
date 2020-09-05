# Helper script for mpirun launcher

# MVAPICH2-GDR. Tested on systems with InfiniBand
mpirun -np 8 -ppn 4 \
	--hostfile hosts \
	-env MV2_SMP_USE_CMA=0 \
	-env MV2_DEBUG_SHOW_BACKTRACE=1 \
	-env MV2_USE_CUDA=1 \
       	-env MV2_ENABLE_AFFINITY=0 \
	-env MV2_SUPPORT_DL=1 \
	sh mpi_train_bert_onebitadam_seq128.sh


