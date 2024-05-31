import os
import argparse
from solver import test_resnet_with_cub

def main():
    parser = argparse.ArgumentParser("Test Fine-tuned Resnet model on the CUB dataset")
    parser.add_argument("--path", type=str, help='The file path of the parameters of model.')
    
    args = parser.parse_args()
    
    test_resnet_with_cub(
        data_dir=os.path.join("data", "CUB_200_2011"),
        path=args.path
    )
    
if __name__ == '__main__':
    main()