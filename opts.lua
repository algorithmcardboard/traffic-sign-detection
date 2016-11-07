local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine();
    cmd:text()
    cmd:text('The German Traffic Sign Recognition Benchmark: A multi-class classification ')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-data',             './data',             'Path to dataset')
    cmd:option('-val',              10,             'Percentage to use for validation set')
    cmd:option('-nEpochs',          2,            'Maximum epochs')
    cmd:option('-batchsize',        128,             'Batch size for epochs')
    cmd:option('-nThreads',         1,              'Number of dataloading threads')
    cmd:option('-LR',               0.1,            'initial learning rate')
    cmd:option('-momentum',         0.9,            'momentum')
    cmd:option('-weightDecay',      1e-4,           'weight decay')
    cmd:option('-logDir',           'logs',         'log directory')
    cmd:option('-name',             '',             'name of the current training run')
    cmd:option('-manualSeed',       30,             'Manually set RNG seed')
    cmd:option('-model',            '',             'Model to use for training')

    local opt = cmd:parse(arg or {})

    if opt.model == '' or not paths.filep('models/'..opt.model..'.lua') then
        cmd:error('Invalid model ' .. opt.model)
    end

    if opt.data == '' or not paths.dirp(opt.data) then
        cmd:error('Invalid data path ' .. opt.data)
    end

    return opt
end

return M
