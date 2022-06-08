# yoshitomo-matsubara/bert-large-uncased-
for task in  mrpc #sst2 stsb mnli qqp rte cola mrpc qnli
do
git lfs clone https://huggingface.co/yoshitomo-matsubara/bert-base-uncased-$task &
done
