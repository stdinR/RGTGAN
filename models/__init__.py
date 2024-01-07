import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image restoration
    if model == 'RGTNet_model':
        from .RGTNet_model import RGTNetModel as M
    elif model == 'RGTGAN_model':
        from .RGTGAN_model import RGTGANModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
