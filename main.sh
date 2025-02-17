for topk in 5; do
    log_file="./log/topk_${topk}.txt"
    python ./main.py -topk $topk | tee $log_file
done 