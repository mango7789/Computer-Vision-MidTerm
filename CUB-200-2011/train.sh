#!/bin/bash

# define the Output directory
output_dir="Output"

# create the output directory if it doesn't exist
mkdir -p "$output_dir"

###############################################################################
##                     find optimal learning rates                           ##
###############################################################################

# Define hyperparameters
num_epochs=15
fine_tuning_lrs=(5e-5 1e-4 5e-4)
output_lrs=(1e-3 2e-3 5e-3 8e-3 1e-2)

# initialize best accuracy and corresponding hyperparameters
best_acc=0
best_fine_tune_lr=""
best_output_lr=""

# initialize list to store accuracies for each configuration
best_accs=()

# loop through all combinations of hyperparameters
for ft_lr in "${fine_tuning_lrs[@]}"; do
    for fc_lr in "${output_lrs[@]}"; do

        # train the model with current hyperparameters
        curr_best_acc=$(python train.py --epochs $num_epochs --ft_lr $ft_lr --fc_lr $fc_lr)
        best_accs+=("$curr_best_acc")

        # check if current accuracy is the best so far
        if (($(echo "$curr_best_acc > $best_acc" | bc -l))); then
            best_acc=$curr_best_acc
            best_fine_tune_lr=$ft_lr
            best_output_lr=$fc_lr
        fi
    done
done

# write results into a .txt file
echo "Configuration        Accuracy" >"$output_dir/best_accuracy_lr.txt"
echo "===============================" >>"$output_dir/best_accuracy_lr.txt"
for ((i = 0; i < ${#fine_tuning_lrs[@]}; ++i)); do
    ft_lr=${fine_tuning_lrs[$i]}
    fc_lr=${output_lrs[$i]}
    accuracy=${best_accs[$i]}
    printf "(%7.5f, %7.5f)   %8.6f\n" $ft_lr $fc_lr $accuracy >>"$output_dir/best_accuracy_lr.txt"
done

echo "Best accuracy: $best_acc"
echo "Best hyperparameters: ($best_fine_tune_lr, $best_output_lr)"

###############################################################################
##                          find optimal epoch                               ##
###############################################################################

# train for different numbers of epochs
num_epochs=(15 30 45)
best_acc=$(python train.py --epochs "${num_epochs[@]}" --ft_lr $best_fine_tune_lr --fc_lr $best_output_lr --save)

# write result into a .txt file
echo "Learning Rate  Accuracy" >"$output_dir/best_accuracy_ep.txt"
echo "======================================" >>"$output_dir/best_accuracy_ep.txt"
printf "%7.5f     %8.6f\n" $lr $best_acc >>"$output_dir/best_accuracy_ep.txt"

###############################################################################
##                    random initialization of weight                        ##
###############################################################################

# random initialization of weight
python train.py --epochs "${num_epochs[@]}" --ft_lr $best_fine_tune_lr --fc_lr $best_output_lr --pretrain False
