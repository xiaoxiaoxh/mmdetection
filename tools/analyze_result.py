import seaborn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def analyze(result, samples_per_cls):
    samples_per_cls = samples_per_cls.cuda()
    if 'fc_weight' in result:
        fc_weight = result['fc_weight'].cuda()
        idxs = torch.argsort(samples_per_cls, descending=True)
        fc_weight = fc_weight[1:, ].t()[:, idxs]
        # fc_weight = fc_weight[1:, ].t()
        fc_weight_normalized = F.normalize(fc_weight, dim=0)
        num_classes = idxs.shape[0]
        corr_mat = torch.empty(num_classes, num_classes).cuda()
        corr_mat_cos = torch.empty(num_classes, num_classes).cuda()
        for i in range(num_classes):
            corr_mat[i, :] = torch.sum(fc_weight[:, i].unsqueeze(1) * fc_weight, dim=0)
            corr_mat_cos[i, :] = torch.sum(fc_weight_normalized[:, i].unsqueeze(1) *
                                           fc_weight_normalized, dim=0)
        corr_mat_cos = corr_mat_cos.cpu().numpy()

        plt.figure(figsize=(13, 10))
        seaborn.heatmap(corr_mat_cos, vmin=-0.5, vmax=0.5, center=0, annot=False, square=False,
                        xticklabels=False, yticklabels=False)

        plt.figure()
        plt.plot(range(num_classes), torch.sort(samples_per_cls, descending=True)[0].cpu().numpy())
        plt.yscale('log')
        plt.grid()

        plt.show()
        print('Finish analyzing!!')
