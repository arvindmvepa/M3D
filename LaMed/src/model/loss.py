import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        target_ = target.clone().float()
        target_[target == -1] = 0
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match\n" + str(
            predict.shape) + '\n' + str(target.shape[0])
        predict = predict.contiguous().view(predict.shape[0], -1)
        target_ = target_.contiguous().view(target_.shape[0], -1)

        num = torch.sum(torch.mul(predict, target_), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target_, dim=1) + self.smooth

        dice_score = 2 * num / den
        dice_loss = 1 - dice_score

        # dice_loss_avg = dice_loss[target[:,0]!=-1].sum() / dice_loss[target[:,0]!=-1].shape[0]
        dice_loss_avg = dice_loss.sum() / dice_loss.shape[0]

        return dice_loss_avg


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match\n' + str(predict.shape) + '\n' + str(
            target.shape)
        target_ = target.clone()
        target_[target == -1] = 0

        ce_loss = self.criterion(predict, target_.float())

        return ce_loss


class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        embeddings = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        labels = torch.arange(batch_size).repeat(2).to(z_i.device)
        mask = torch.eye(batch_size * 2).bool().to(z_i.device)
        similarity_matrix = similarity_matrix[~mask].view(batch_size * 2, -1)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


class MultiModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features_list):
        batch_size = features_list[0].shape[0]
        num_modalities = len(features_list)
        device = features_list[0].device

        normalized_features = [F.normalize(features, dim=1) for features in features_list]

        total_loss = 0
        num_pairs = 0

        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                z_i = normalized_features[i]
                z_j = normalized_features[j]

                similarity = torch.matmul(z_i, z_j.T) / self.temperature
                labels = torch.arange(batch_size, device=device)

                loss_i = F.cross_entropy(similarity, labels)
                loss_j = F.cross_entropy(similarity.T, labels)

                total_loss += (loss_i + loss_j) / 2
                num_pairs += 1

        return total_loss / num_pairs


class GroupContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def create_group_mask(self, labels):
        batch_size = labels.shape[0]
        mask = torch.all(labels.unsqueeze(1) == labels.unsqueeze(0), dim=2)
        return mask.float()

    def forward(self, features_list, labels):
        batch_size = features_list[0].shape[0]
        group_mask = self.create_group_mask(labels)
        normalized_features = [F.normalize(features, dim=1) for features in features_list]

        total_loss = 0
        num_pairs = 0

        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                z_i = normalized_features[i]
                z_j = normalized_features[j]

                similarity = torch.matmul(z_i, z_j.T) / self.temperature

                exp_similarity = torch.exp(similarity)
                pos_sum = torch.sum(exp_similarity * group_mask, dim=1)
                neg_sum = torch.sum(exp_similarity * (1 - group_mask), dim=1)

                loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
                total_loss += loss.mean()
                num_pairs += 1

        return total_loss / num_pairs
