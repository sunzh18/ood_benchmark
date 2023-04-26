import argparse
import json


def modify_command_options(opts):


    return opts


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="resnet18", help="Which network to use: resnet18, resnet50, mobilenet, BiT-S-R101x1, vit_base_patch16_384")
    parser.add_argument('--in_dataset', default="CIFAR-10", type=str, help='in-distribution dataset, CIFAR-10, CIFAR-100, imagenet')
    parser.add_argument('--out_dataset', default=None, type=str, help="['SVHN', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365']  ['inat', 'sun50', 'places50', 'dtd', ]")

    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")
    parser.add_argument("--logdir", type=str, default="result",
                        help="Where to log test info (small).")
    parser.add_argument("--name", type=str, required=True,
                        help="Name of method. Used for monitoring and checkpointing. baseline, BATS, my")
    parser.add_argument('--score', default='energy', type=str, help='score function,  odin mahalanobis CE_with_Logst MSP, Energy, KL_Div')


    # parser.add_argument("--in_datadir", help="Path to the in-distribution data folder.")
    # parser.add_argument("--out_datadir", help="Path to the out-of-distribution data folder.")

    parser.add_argument("--arch", default=None ,type=str, help="network arch") 
    parser.add_argument('--num_classes', default=10, type=int, help='number of class')
    parser.add_argument("--batch", type=int, default=256,
                        help="Batch size.")
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='number of learning rate')
    parser.add_argument("--model_path", default=None ,type=str, help="Path to the finetuned model you want to test")  
    parser.add_argument("--validation", default=False, help="use validation set")  
    parser.add_argument("--test", default=False, help="test acc")
    parser.add_argument("--wandb", default=None, type=str)
    
    # parser.add_argument('--name', default="densenet", type=str,
    #                     help='neural network name and training set')
    # parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')


    # parser.add_argument('--epsilon', default=8, type=int, help='epsilon')

    parser.add_argument('--threshold', default=1.0, type=float, help='sparsity level')
    parser.add_argument('--gpu', default='0', type=str, help='gpu index')
    parser.add_argument('--bats', default=0, type=int, help='Using BATS to boost the performance or not.')

    # arguments for ODIN
    parser.add_argument('--temperature_odin', default=1000, type=int,
                        help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.0014, type=float,
                        help='perturbation magnitude for odin')

    # arguments for Energy
    parser.add_argument('--temperature_energy', default=1, type=int,
                        help='temperature scaling for energy')

    # arguments for GradNorm
    parser.add_argument('--temperature_gradnorm', default=1, type=int,
                        help='temperature scaling for GradNorm')

    # arguments for mahalanobis
    parser.add_argument('--mahalanobis_param_path', default=None, help='path to tuned mahalanobis parameters')

    
    # parser.set_defaults(argument=True)



    # parser.add_argument("--diff_model", type=str, default='diff_model4')
    # parser.add_argument("--diff_result", type=str, default='diff_result4')
    # Method Options
    # BE CAREFUL USING THIS, THEY WILL OVERRIDE ALL THE OTHER PARAMETERS.

    # Train Options
#     parser.add_argument("--epochs", type=int, default=30, help="epoch number (default: 30)")
#     parser.add_argument("--batch_size", type=int, default=4, help='batch size (default: 4)')
#     parser.add_argument("--crop_size", type=int, default=512, help="crop size (default: 513)")

    


    return parser
