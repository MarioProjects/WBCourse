import argparse


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(
    description='Moving Beyond Baseline', formatter_class=SmartFormatter
)

parser.add_argument(
    '--lr', type=float, default=0.01, help='R|Learning rate'
)

parser.add_argument(
    '--batch_size', type=int, default=8, help='R|Batch size'
)

parser.add_argument(
    '--num_classes', type=int, default=1, help='R|Number of classes'
)

parser.add_argument(
    '--num_epochs', type=int, default=15, help='R|Number of epochs'
)

parser.add_argument(
    '--optimizer', type=str, default='SGD', help='R|Optimizer for training'
)

parser.add_argument(
    '--scheduler', type=str, default='MultiStepLR', help='R|Scheduler for optimizer'
)

parser.add_argument(
    '--milestones', type=list, default=[5, 10], help='R|Milestones for scheduler'
)

parser.add_argument(
    '--gamma', type=float, default=0.1, help='R|Gamma for scheduler'
)

parser.add_argument(
    '--model', type=str, default='resnet18', help='R|Model'
)

parser.add_argument(
    '--pretrained', type=bool, default=False, help='R|Pretrained model'
)

parser.add_argument(
    '--loss', type=str, default='BCEWithLogitsLoss', help='R|Loss'
)

config = parser.parse_args()