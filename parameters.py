import argparse

def get_parameters():
    parser = argparse.ArgumentParser(description="Hyperparameters for ResNet18 training")

    # Training related
    parser.add_argument('--num_epoch', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for optimizer')
    parser.add_argument('--valid_size', type=float, default=0.2, help='Proportion of training data used for validation')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')

    # Model related
    parser.add_argument('--model_save_dir', type=str, default='./model_weights/', help='Directory to save model weights')

    # Device related
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training and testing')

    # Data related
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for dataset storage')

    return parser.parse_args()

if __name__ == "__main__":
    params = get_parameters()
    print(params)