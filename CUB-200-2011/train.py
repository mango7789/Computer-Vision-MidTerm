import os
import argparse
from solver import train_resnet_with_cub

def main():
    parser = argparse.ArgumentParser("Train Resnet on the CUB dataset.")
    
    parser.add_argument("--epochs", type=int, nargs='+', default=[30], help='Training epoch of the model.')
    parser.add_argument("--ft_lr", type=float, default=1e-4, help='The fine-tuning learning rate of the Resnet-18 except the fc layer.')
    parser.add_argument('--fc_lr', type=float, default=1e-3, help='The learning rate of the fully-connect linear layer.')
    parser.add_argument('--pretrain', type=bool, default=True, help='The Resnet-18 should use pretrained weight or not, default is True.')
    parser.add_argument('--save', type=bool, default=False, help='The trained best model should be saved or not, default is False.')
    parser.add_argument('--seed', type=int, default=42, help='The random seed of the model.')
    parser.add_argument('--momentum', type=float, default=0.9, help='The momentum of the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='The weight decay of the weight matrices of the model.')
    parser.add_argument('--step', type=int, default=30, help='The step size of learning rate decay.')
    parser.add_argument('--gamma', type=float, default=0.1, help='The scale of learning rate decay of the model.')
    
    args = parser.parse_args()
    
    train_resnet_with_cub(
        data_dir=os.path.join("data", "CUB_200_2011"),
        num_epochs=args.epochs,
        fine_tuning_lr=args.ft_lr,
        output_lr=args.fc_lr,
        pretrain=args.pretrain,
        save=args.save,
        seed=args.seed,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        step=args.step,
        gamma=args.gamma
    )
    
if __name__ == '__main__':
    main()
    