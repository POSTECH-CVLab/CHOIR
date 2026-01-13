import argparse


def get_config():
    parser = argparse.ArgumentParser()

    # Experiment setup
    parser.add_argument("-n", "--name", type=str, help="experiment name")
    parser.add_argument("--task", type=str, default="orientation", help="task name")
    parser.add_argument("--version", type=int, default=1, help="version 1, 2")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--num_gpus", type=int, default=1, help="# gpus to use")
    parser.add_argument("--not_use_tf32", action="store_false", help="allow tf32 matmul or not")

    # Augmentation
    parser.add_argument("--use_kpr", action="store_true", help="knn patch removal")
    parser.add_argument("--kpr_k", type=int, default=100, help="k for knn patch removal")
    parser.add_argument(
        "--num_patches", type=int, default=1, help="# patches for knn patch removal"
    )

    parser.add_argument("--use_gn", action="store_true", help="Gaussian noise")
    parser.add_argument("--noise_amp", type=float, default=0.025, help="noise amplitude")

    parser.add_argument("--num_points", type=int, default=1024, help="# points to sample")
    parser.add_argument(
        "--sample_alg", type=str, default="random", choices=["fixed", "random", "fps"],
        help="which algorithm to use for point sampling"
    )

    # Dataset
    parser.add_argument("--dataset", default="shapenet", help="dataset")
    parser.add_argument(
        "--label_in_use", type=str, default=None, help="labels to use (None for all classes)"
    )
    parser.add_argument("--cross_p", type=float, default=0.5, help="probability to sample the other instance")
    parser.add_argument("--num_rotations", type=int, default=4, help="# rotations for stability")
    parser.add_argument("--shapenet_path", type=str, default="data/shapenet", help="ShapeNet path")
    parser.add_argument("--resample", action="store_true", help="resample during test-time")

    # Dataloader
    parser.add_argument("--batch_size", type=int, default=32, help="batch size") # 24GB limit
    parser.add_argument("--num_workers", type=int, default=8, help="# workers")

    # Model [Backbone]
    parser.add_argument("--backbone", type=str, default="vnt", help="backbone name")
    parser.add_argument("--vnt_k", type=int, default=40, help="k for knn within the model")
    parser.add_argument("--vnt_pooling", type=str, default="max", choices=["max", "mean"], help="pooling type")
    parser.add_argument("--vnt_base_ch", type=int, default=64, help="# channels of VNT")
    parser.add_argument("--vnt_nlatent", type=int, default=1020, help="# channels of VNT")
    parser.add_argument("--vnt_not_use_ft", action="store_false", help="use feature transform in VNT or not")
    parser.add_argument(
        "--vnt_norm", type=str, default="norm", choices=["norm", "softmax"],
        help="normalization type of VNT (default: norm)"
    )

    # Model [Head]
    parser.add_argument(
        "--head", type=str, default="twostream-inv-naive-pt", help="orientation head name"
    ) # "twostream-inv-naive-pt" == "ours"
    parser.add_argument("--pt_mid_channels", type=int, default=64, help="mid channels of Point Transformer")
    parser.add_argument("--pt_num_blocks", type=int, default=4, help="# blocks of Point Transformer")

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer name")
    parser.add_argument("--lr", type=float, default=0.01, help="the learning rate")
    parser.add_argument("--num_epochs", type=int, default=1500, help="training epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="l2 regularization")

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine", help="learning rate scheduler")

    # Misc
    parser.add_argument("--val_freq", type=int, default=10, help="validation frequency (epoch)")
    parser.add_argument("--log_freq", type=int, default=50, help="training log frequency (step)")

    return parser.parse_args()