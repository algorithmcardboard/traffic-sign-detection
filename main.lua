require 'torch'
require 'optim'
require 'os'
-- require 'cunn'

local optParser = require 'opts'
local tnt = require 'torchnet'
local opt = optParser.parse(arg)


torch.setdefaulttensortype('torch.DoubleTensor')

-- torch.setnumthreads(1)
torch.manualSeed(opt.manualSeed)
-- cutorch.manualSeedAll(opt.manualSeed)

