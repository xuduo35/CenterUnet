import torch.nn as nn

from . import base
from . import functional as F
from  .base import Activation
from . import lovasz_losses as L

class JaccardLoss(base.Loss):

    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, 1., activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass

class LovaszHingeLoss(BCEDiceLoss):
    __name__ = 'lovasz_hinge_loss'

    def __init__(self, eps=1e-7, activation='sigmoid', withfocalloss=False, alpha=0.25, gamma=2, OHEM_percent=0.005):
        super().__init__(eps, activation)
        self.withfocalloss =  withfocalloss
        self.alpha = alpha
        self.gamma = gamma
        self.OHEM_percent = OHEM_percent

    def focal_loss(self, output, target, alpha, gamma, OHEM_percent):
        output = output.contiguous().view(-1)
        target = target.contiguous().view(-1)

        max_val = (-output).clamp(min=0)
        loss = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = nn.functional.logsigmoid(-output * (target * 2 - 1))
        focal_loss = alpha * (invprobs * gamma).exp() * loss

        # Online Hard Example Mining: top x% losses (pixel-wise).
        # Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
        OHEM, _ = focal_loss.topk(k=int(OHEM_percent * [*focal_loss.shape][0]))

        return OHEM.mean()

    def forward(self, y_pr, y_gt):
        bcedice = super().forward(y_pr, y_gt)
        #lovasz_hinge = L.lovasz_hinge(y_pr, y_gt, ignore=255)
        #lovasz_hinge = L.lovasz_hinge_elu(y_pr, y_gt, ignore=255)
        lovasz_hinge = L.lovasz_hinge_elu(y_pr, y_gt, per_image=True, ignore=255)

        if not self.withfocalloss:
            return bcedice + lovasz_hinge

        return bcedice + lovasz_hinge + self.focal_loss(y_pr, y_gt, self.alpha, self.gamma, self.OHEM_percent)

class LovaszHingeLossSymmetric(BCEDiceLoss):
    __name__ = 'lovasz_hinge_loss_symmetric'

    def __init__(self, eps=1e-7, activation='sigmoid', withfocalloss=False, alpha=0.25, gamma=2, OHEM_percent=0.005):
        super().__init__(eps, activation)
        self.withfocalloss =  withfocalloss
        self.alpha = alpha
        self.gamma = gamma
        self.OHEM_percent = OHEM_percent

    def focal_loss(self, output, target, alpha, gamma, OHEM_percent):
        output = output.contiguous().view(-1)
        target = target.contiguous().view(-1)

        max_val = (-output).clamp(min=0)
        loss = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = nn.functional.logsigmoid(-output * (target * 2 - 1))
        focal_loss = alpha * (invprobs * gamma).exp() * loss

        # Online Hard Example Mining: top x% losses (pixel-wise).
        # Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
        OHEM, _ = focal_loss.topk(k=int(OHEM_percent * [*focal_loss.shape][0]))

        return OHEM.mean()

    def forward(self, y_pr, y_gt):
        bcedice = super().forward(y_pr, y_gt)
        #lovasz_hinge = L.lovasz_hinge(y_pr, y_gt, ignore=255)
        #lovasz_hinge = (L.lovasz_hinge(y_pr, y_gt, ignore=255) + L.lovasz_hinge(-y_pr, 1-y_gt, ignore=255))/2
        #lovasz_hinge = (L.lovasz_hinge_elu(y_pr, y_gt, ignore=255) + L.lovasz_hinge_elu((-y_pr), (1-y_gt), ignore=255))/2
        lovasz_hinge = (L.lovasz_hinge_elu(y_pr, y_gt, per_image=True, ignore=255) + L.lovasz_hinge_elu((-y_pr), (1-y_gt), per_image=True, ignore=255))/2

        if not self.withfocalloss:
            return bcedice + lovasz_hinge

        return bcedice + lovasz_hinge + self.focal_loss(y_pr, y_gt, self.alpha, self.gamma, self.OHEM_percent)
