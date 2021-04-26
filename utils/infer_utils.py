import torch



def decode_infer(output, stride):
    # logging.info(torch.tensor(output.shape[0]))
    # logging.info(output.shape)
    # # bz is batch-size
    # bz = tuple(torch.tensor(output.shape[0]))
    # gridsize = tuple(torch.tensor(output.shape[-1]))
    # logging.info(gridsize)
    sh = torch.tensor(output.shape)
    bz = sh[0]
    gridsize = sh[-1]

    output = output.permute(0, 2, 3, 1)
    output = output.view(bz, gridsize, gridsize, self.gt_per_grid, 5+self.numclass)
    x1y1, x2y2, conf, prob = torch.split(
        output, [2, 2, 1, self.numclass], dim=4)

    shiftx = torch.arange(0, gridsize, dtype=torch.float32)
    shifty = torch.arange(0, gridsize, dtype=torch.float32)
    shifty, shiftx = torch.meshgrid([shiftx, shifty])
    shiftx = shiftx.unsqueeze(-1).repeat(bz, 1, 1, self.gt_per_grid)
    shifty = shifty.unsqueeze(-1).repeat(bz, 1, 1, self.gt_per_grid)

    xy_grid = torch.stack([shiftx, shifty], dim=4).cuda()
    x1y1 = (xy_grid+0.5-torch.exp(x1y1))*stride
    x2y2 = (xy_grid+0.5+torch.exp(x2y2))*stride

    xyxy = torch.cat((x1y1, x2y2), dim=4)
    conf = torch.sigmoid(conf)
    prob = torch.sigmoid(prob)
    output = torch.cat((xyxy, conf, prob), 4)
    output = output.view(bz, -1, 5+self.numclass)
    return output