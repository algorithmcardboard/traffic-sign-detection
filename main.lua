require 'torch'
require 'optim'
require 'os'
require 'optim'
-- require 'cunn'

local optParser = require 'opts'
local opt = optParser.parse(arg)


print('opt is ', opt)


local tnt = require 'torchnet'
local model = require 'models/'.. opt.model

local DATA_PATH = '/home/anirudhan/workspace/traffic-sign-detection/data/'
local WIDTH, HEIGHT = 32, 32


torch.setdefaulttensortype('torch.DoubleTensor')

-- torch.setnumthreads(1)
torch.manualSeed(opt.manualSeed)
-- cutorch.manualSeedAll(opt.manualSeed)

function resize(img)
    return image.scale(img, WIDTH,HEIGHT)
end

function transformInput(inp)
    f = tnt.transform.compose{
        [1] = resize
    }
    return f(inp)
end

function getTrainSample(dataset, idx)
    r = dataset[idx]
    classId, track, file = r[9], r[1], r[2]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    return transformInput(image.load(DATA_PATH .. '/train_images/'..file))
end

function getTrainLabel(dataset, idx)
    return dataset[idx][9]
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    return transformInput(image.load(file))
end

function getIterator(dataset)
    return tnt.ParallelDatasetIterator{
        nthreads = opt.nthreads,
        init = function() require 'torchnet' end,
        closure = function()
            return tnt.BatchDataset{
                batchsize = opt.batchsize,
                dataset = dataset
            }
        end
    }
end

trainDataset = tnt.SplitDataset{
    partitions = {train=0.9, val=0.1},
    dataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
            list = torch.range(1, train:size(1)):long(),
            load = function(idx)
                return {
                    input =  getTrainSample(train, idx),
                    target = getTrainLabel(train, idx)
                }
            end
        }
    }
}

testDataset = tnt.ListDataset{
    list = torch.range(1, test:size(1)):long(),
    load = function(idx)
        return {
            input = getTestSample(test, idx)
        }
    end
}


local engine = tnt.OptimEngine()
local meter = tnt.ClassErrorMeter{topk = {1}}

engine.hooks.onStartEpoch = function(state)
    meter:reset()
    clerr:reset()
end

engine:train{
    network = model,
    criterion = criterion,
    iterator = iterator,
    optimMethod = optim.sgd,
    config = {
        learningRate = opt.learningRate,
        momentum = opt.momentum
    }
}
