import torch.nn as nn

'''
This model named cyle_gan, which is used to transform the image's style

Input: origin image 
Output: style-transformed image  

Generator G_A: fake_B = G_A(A)
Generator G_B: fake_A = G_B(B)

Discriminator D_A: D_A(B) vs D_A(fake_B)
Discriminator D_B: D_B(A) vs D_B(fake_A)

loss D_A: loss of Discriminator D_A
loss G_A: loss of Generator G_A ==> D_A 
loss cycle_A: lambda_A * ||G_B(G_A(A)) - A||
loss idt_A: lambda_identity * ||G_A(B) - B|| * lambda_B

loss D_B: loss of Discriminator D_B
loss G_B: loss of Generator G_B ==> D_B
loss cycle_B: lambda_B * ||G_A(G_B(B)) - B||
loss idt_B: lambda_identity * ||G_B(A) - A|| * lambda_A

'''

class CycleGan(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        self.direction = cfg.INPUT.DIRECTION
        self.is_train = self.cfg.TRAIN.IS_TRAIN:
        if self.is_train:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B'] 
        else:
            self.model_names = ['G_A', 'G_B']

    def forward(self, x):