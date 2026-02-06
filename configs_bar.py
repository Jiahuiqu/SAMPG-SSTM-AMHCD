import os


class Configs:
    model_name = 'DADSSFF'
    device = '0'
    random_seed = 42
    bx = 'affine'
    """**************** dataset and directory ****************"""
    dataset_name = 'Barbara'  # China, Hermiston, River, BayArea, Barbara, Liyukou
    patch_size = 9

    """**************** Hyper Arguments for Training ****************"""
    in_feats = 224
    hi_feats = 256
    iter_num = 3
    train_batch_size = 32
    val_batch_size = 1000
    epochs = 100
    warmup_epochs = 5
    train_num_workers = 12
    val_num_workers = 8
    save_ckpt_interval = 50
    clip_grad = 0.3
    is_resume_model = False
    use_clip_grad = False

    """**************** Hyper Arguments for Testing ****************"""
    test_batch_size = 1000
    test_num_workers = 8

    """**************** Hyper Arguments for Optimizer ****************"""
    # 学习率：小的学习率收敛慢，但能将loss值降到更低。当使用平方和误差作为成本函数时，随着数据量的增多，学习率应该被设置为相应更小的值。
    # adam一般0.001，sgd0.1，batch_size增大n倍，学习率一般也要增大根号n倍
    # weight_decay:通常1e-4 —— 1e-5，值越大表示正则化越强。数据集大、复杂，模型简单，调小；数据集小模型越复杂，调大。
    lr = 0.001
    min_lr = 34e-6
    weight_decay = 0.001  # for SGD and Adam
    eps = 1e-4
    momentum = 0.9  # for SGD
    # step scheduler
    lr_step = 35  # step
    milestones = [35, 70]  # multistep
    lr_gamma = 0.5

    """**************** Log and Save Folder Path ****************"""
    log_path = 'logs/' + model_name + '_' + dataset_name
    save_ckpt_folder = 'checkpoints/' + model_name + '_' + dataset_name + f"/epoch{epochs}_lr{lr}_" \
                                                                          f"batchsize{train_batch_size}_" \
                                                                          f"patchsize{patch_size}"
    resume_path = 'checkpoints/' + model_name + '_' + dataset_name + \
                  f'/epoch{epochs}_lr{lr}_batchsize{train_batch_size}_patchsize{patch_size}' + \
                  f'/{bx}/best_acc.pth'
    save_results_folder = 'results/' + model_name + '_' + dataset_name + f"/epoch{epochs}_lr{lr}_" \
                                                                         f"batchsize{train_batch_size}_" \
                                                                         f"patchsize{patch_size}"


if __name__ == '__main__':
    # if not os.path.exists(os.path.join(os.getcwd(), Configs.save_folder)):
    #     os.makedirs(Configs.save_folder)
    print(os.path.exists(os.path.join(Configs.save_ckpt_folder)))
