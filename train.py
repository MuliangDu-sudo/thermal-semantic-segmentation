from utils import AverageMeter, set_requires_grad

# TODO: add predict function
def predict(args):
    foo = []
    return foo


def train(args, s_data, t_data, g_s2t, g_t2s, d_s, d_t, gan_loss_func, cycle_loss_func, identity_loss_func, sem_loss_func,
          optim_g, optim_d, fake_s_pool, fake_t_pool):
    """
    :param args: parser
    :param s_data: source train dataloader
    :param t_data: target train dataloader
    :param g_s2t: source to target generator
    :param g_t2s: target to source generator
    :param d_s: source discriminator
    :param d_t: target discriminator
    :param gan_loss_func: gan loss function
    :param cycle_loss_func: reconstruction loss function
    :param identity_loss_func: identity loss function
    :param sem_loss_func: semantic consistency loss function
    :param optim_g: generator optimizer
    :param optim_d: discriminator optimizer
    :param fake_s_pool: source image pool for discriminator
    :param fake_t_pool: target image pool for discriminator
    :return:
    """

    losses_g_s2t = AverageMeter('g_s2t', ':3.2f')
    losses_g_t2s = AverageMeter('g_t2s', ':3.2f')
    losses_d_s = AverageMeter('d_s', ':3.2f')
    losses_d_t = AverageMeter('d_t', ':3.2f')
    losses_cycle_s = AverageMeter('cycle_s', ':3.2f')
    losses_cycle_t = AverageMeter('cycle_t', ':3.2f')
    losses_identity_s = AverageMeter('idt_s', ':3.2f')
    losses_identity_t = AverageMeter('idt_t', ':3.2f')
    losses_semantic_s2t = AverageMeter('sem_s2t', ':3.2f')
    losses_semantic_t2s = AverageMeter('sem_t2s', ':3.2f')

    # TODO: ProgressMeter: for visualizer

    for real_s, label_s, real_t, _ in zip(s_data, t_data):
        real_s = real_s.to(args.device)
        real_t = real_t.to(args.device)
        label_s = label_s.to(args.device)

        # forward pass
        fake_t = g_s2t(real_s)
        rec_s = g_t2s(fake_t)
        fake_s = g_t2s(real_t)
        rec_t = g_s2t(fake_s)

        # Optimizing generators
        # discriminators require no gradients
        set_requires_grad(d_s, False)
        set_requires_grad(d_t, False)

        optim_g.zero_grad()
        # gan loss d_s(g_s2t(s))
        loss_g_s2t = gan_loss_func(d_t(fake_t), real=True)
        # GAN loss d_S(g_t2s(T))
        loss_g_t2s = gan_loss_func(d_s(fake_s), real=True)
        # Cycle loss || g_t2s(g_s2t(s)) - s||
        loss_cycle_s = cycle_loss_func(rec_s, real_s) * args.trade_off_cycle
        # Cycle loss || g_s2t(g_t2s(t)) - t||
        loss_cycle_t = cycle_loss_func(rec_t, real_t) * args.trade_off_cycle
        # Identity loss
        # g_s2t should be identity if real_t is fed: ||g_s2t(real_t) - real_t||
        identity_t = g_s2t(real_t)
        loss_identity_t = identity_loss_func(identity_t, real_t) * args.trade_off_identity
        # g_t2s should be identity if real_s is fed: ||g_t2s(real_s) - real_s||
        identity_s = g_t2s(real_s)
        loss_identity_s = identity_loss_func(identity_s, real_s) * args.trade_off_identity
        # Semantic loss
        pred_fake_t = predict(fake_t)
        pred_real_s = predict(real_s)
        loss_semantic_s2t = sem_loss_func(pred_fake_t, label_s) * args.trade_off_semantic
        pred_fake_s = predict(fake_s)
        pred_real_t = predict(real_t)
        loss_semantic_t2s = sem_loss_func(pred_fake_s, pred_real_t.max(1).indices) * args.trade_off_semantic
        # combined loss and calculate gradients
        loss_g = loss_g_s2t + loss_g_t2s + loss_cycle_s + loss_cycle_t + loss_identity_s + loss_identity_t + \
                 loss_semantic_s2t + loss_semantic_t2s
        loss_g.backward()
        optim_g.step()

        # Optimize discriminator
        set_requires_grad(d_s, True)
        set_requires_grad(d_t, True)
        optim_d.zero_grad()
        # Calculate GAN loss for discriminator D_S
        fake_s_ = fake_s_pool.query(fake_s.detach())
        loss_d_s = 0.5 * (gan_loss_func(d_s(real_s), True) + gan_loss_func(d_s(fake_s_), False))
        loss_d_s.backward()
        # Calculate GAN loss for discriminator D_T
        fake_t_ = fake_t_pool.query(fake_t.detach())
        loss_d_t = 0.5 * (gan_loss_func(d_t(real_t), True) + gan_loss_func(d_t(fake_t_), False))
        loss_d_t.backward()
        optim_d.step()

        # measure elapsed time
        losses_g_s2t.update(loss_g_s2t.item(), real_s.size(0))
        losses_g_t2s.update(loss_g_t2s.item(), real_s.size(0))
        losses_d_s.update(loss_d_s.item(), real_s.size(0))
        losses_d_t.update(loss_d_t.item(), real_s.size(0))
        losses_cycle_s.update(loss_cycle_s.item(), real_s.size(0))
        losses_cycle_t.update(loss_cycle_t.item(), real_s.size(0))
        losses_identity_s.update(loss_identity_s.item(), real_s.size(0))
        losses_identity_t.update(loss_identity_t.item(), real_s.size(0))
        losses_semantic_s2t.update(loss_semantic_s2t.item(), real_s.size(0))
        losses_semantic_t2s.update(loss_semantic_t2s.item(), real_s.size(0))

        # TODO: add visualizer


