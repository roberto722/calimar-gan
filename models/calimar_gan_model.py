import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.amp import GradScaler

class CalimarGanModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0)
            parser.add_argument('--lambda_B', type=float, default=10.0)
            parser.add_argument('--lambda_identity', type=float, default=0.5)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.isTrain = opt.isTrain

        if self.isTrain:
            self.lambda_idt = opt.lambda_identity
            self.lambda_A = opt.lambda_A
            self.lambda_B = opt.lambda_B

        self.visual_names = ['real_B', 'fake_A'] if not self.isTrain else ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        self.model_names = ['G_A', 'G_B'] + (['D_A', 'D_B'] if self.isTrain else [])

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'calimar_gan_A', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'calimar_gan_B', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.extend([self.optimizer_G, self.optimizer_D])

            self.scaler = GradScaler('cuda')

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.attention = input['attention'].to(self.device).repeat(1, 3, 1, 1)
        self.real_A = torch.cat((input['A' if AtoB else 'B'].to(self.device), self.attention), dim=1)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B, self.o1_b, self.o2_b, self.o3_b, self.o4_b, self.o5_b, self.o6_b, self.o7_b, self.o8_b, self.o9_b, self.o10_b, \
        self.a1_b, self.a2_b, self.a3_b, self.a4_b, self.a5_b, self.a6_b, self.a7_b, self.a8_b, self.a9_b, self.a10_b, \
        self.i1_b, self.i2_b, self.i3_b, self.i4_b, self.i5_b, self.i6_b, self.i7_b, self.i8_b, self.i9_b = self.netG_A(self.real_A)  # G_A(A)
        
        if self.isTrain:
            self.rec_A, *_ = self.netG_B(self.fake_B)

        self.fake_A, self.o1_a, self.o2_a, self.o3_a, self.o4_a, self.o5_a, self.o6_a, self.o7_a, self.o8_a, self.o9_a, self.o10_a, \
        self.a1_a, self.a2_a, self.a3_a, self.a4_a, self.a5_a, self.a6_a, self.a7_a, self.a8_a, self.a9_a, self.a10_a, \
        self.i1_a, self.i2_a, self.i3_a, self.i4_a, self.i5_a, self.i6_a, self.i7_a, self.i8_a, self.i9_a = self.netG_B(self.real_B) # G_B(B)

        if self.isTrain:
            self.input_A = torch.cat((self.fake_A, self.a1_a), dim=1)
            self.rec_B, *_ = self.netG_A(self.input_A)

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        loss_D = (self.criterionGAN(pred_real, True) + self.criterionGAN(pred_fake, False)) * 0.5
        self.scaler.scale(loss_D).backward()
        return loss_D

    def backward_D_A(self):
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B_pool.query(self.fake_B))

    def backward_D_B(self):
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A[:, :3, :, :], self.fake_A_pool.query(self.fake_A))

    def backward_G(self):
        if self.lambda_idt > 0:
            self.input_idt_backward = torch.cat((self.real_B, self.a1_a), dim=1)
            idt_A, *_ = self.netG_A(self.input_idt_backward)
            self.loss_idt_A = self.criterionIdt(idt_A, self.real_B) * self.lambda_B * self.lambda_idt

            idt_B, *_ = self.netG_B(self.real_A[:, :3, :, :])
            self.loss_idt_B = self.criterionIdt(idt_B, self.real_A[:, :3, :, :]) * self.lambda_A * self.lambda_idt
        else:
            self.loss_idt_A = self.loss_idt_B = 0

        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A[:, :3, :, :]) * self.lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.lambda_B

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.scaler.scale(self.loss_G).backward()

    def optimize_parameters(self):
        with torch.autocast(device_type="cuda"):
            self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.scaler.step(self.optimizer_G)

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.scaler.step(self.optimizer_D)

        self.scaler.update()
