Author: lichen_zeng@sina.cn
Date: 20180913
Subject: Note some key points


1, 一个精巧的 类调用 方法，值得学习
class MNIST(object):
    def __init__(self, batch_size, use_gpu, num_workers):
        pass

__factory = {
    'mnist': MNIST,
}

def create(name, batch_size, use_gpu, num_workers):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](batch_size, use_gpu, num_workers)


2, 重新定义 stdout 的这个操作也很有用，可以用来捕获日志等。
    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))


3, 可以通过这个方式来逐步降低 模型的学习率。
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

    if args.stepsize > 0: scheduler.step()


4, 可以通过 累积坐标值 来增强显示效果。
    all_features, all_labels = [], []

        all_features.append(features.data.cpu().numpy())
        all_labels.append(labels.data.cpu().numpy())

    all_features = np.concatenate(all_features, 0)
    all_labels = np.concatenate(all_labels, 0)
    plot_features(all_features, all_labels, num_classes, epoch, prefix='train')
