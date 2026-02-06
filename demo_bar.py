import os

import torch.autograd
from torch import optim, nn
from configs_bar import Configs as cfgs
from MYModel import SST_MCDNet
from utils.trainer import Trainer
from utils.evaluator_bar import Evaluator
from utils.seed import set_seed

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfgs.device

    set_seed(cfgs.random_seed)
    print(f"seed: {cfgs.random_seed}, hi_feats: {cfgs.hi_feats}, iter_num: {cfgs.iter_num},"
          f" batch_size: {cfgs.train_batch_size}, lr: {cfgs.lr}, patch_size: {cfgs.patch_size},"
          f" momentum: {cfgs.momentum}, weight_decay: {cfgs.weight_decay}.")
    model = SST_MCDNet(in_ch=cfgs.in_feats).cuda()
    print(f"model params: {(sum(p.numel() for p in model.parameters()) / 1e6):.2f}M")
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=cfgs.lr, weight_decay=cfgs.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfgs.milestones, gamma=cfgs.lr_gamma)
    trainer = Trainer(model, criterion, optimizer, scheduler, cfgs, bx='affine')
    trainer.train()
    evaluator = Evaluator(model, cfgs, bx='affine')
    evaluator.evaluate()

if __name__ == '__main__':
    main()
