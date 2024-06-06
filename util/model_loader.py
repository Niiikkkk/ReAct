import torch
from models.resnet import resnet18, resnet50, resnet18_cifar, resnet50_cifar
from models.mobilenet import *


def get_model(args, num_classes, load_ckpt=True):
    if args.in_dataset == 'imagenet':
        if args.model_arch == 'resnet18':
            model = resnet18(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'resnet50':
            model = resnet50(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'mobilenet':
            model = mobilenet_v2(num_classes=num_classes, pretrained=True)
    else:
        if args.model_arch == 'resnet18':
            model = resnet18_cifar(num_classes=num_classes, method=args.method)
        elif args.model_arch == 'resnet50':
            model = resnet50_cifar(num_classes=num_classes, method=args.method)
        else:
            assert False, 'Not supported model arch: {}'.format(args.model_arch)

        if load_ckpt:
            checkpoint = torch.load("./checkpoints/{in_dataset}/{model_arch}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, model_arch=args.name, epochs=args.epochs))
            model.load_state_dict(checkpoint['state_dict'])

    model.cuda()
    model.eval()
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    return model
