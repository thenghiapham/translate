require 'torch'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'fbcunn'


require 'optim'
require 'util.table_utils'

require 'util.pepperfish_profiler'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
-- model params
cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-source_vocab_size',10001,'number of words in the source vocab')
cmd:option('-target_vocab_size',10002,'number of words in the target vocab')
cmd:option('-rnn_size', 50, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
-- optimization
cmd:option('-use_hsm',1,'hierarchial softmax. 0 = use normal softmax. 1 = use hsm in fbcunn')
cmd:option('-cluster_count',100,'number of clusters used for hsm')
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-max_seq_length',100,'maximum sequence length')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size',0,'number of sequences to train on in parallel')
cmd:option('-max_epochs',2,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()
  
-- parse input params
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

local TranslationModel = require 'model.TranslationModel'
local translation_model = TranslationModel.create_model(opt)
local params = translation_model.params
local grad_params = translation_model.grad_params

local PairLoader = require 'util.pair_loader'
local Vocab = require 'util.preprocessing.vocab'

local dir = "/home/nghia/Downloads/fr-en/"
local en_shuffled_file = dir .. "europarl-v7.fr-en.shuf.en"
local fr_shuffled_file = dir .. "europarl-v7.fr-en.shuf.fr"

local loader = PairLoader.create(fr_shuffled_file,en_shuffled_file, opt.target_vocab_size - 2, opt.max_seq_length )


local function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    
    local source, in_target, out_target = loader:next_batch()
    
    
    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        source = source:float():cuda()
        in_target = in_target:float():cuda()
        out_target = out_target:float():cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        source = source:cl()
        in_target = in_target:cl()
        out_target = out_target:cl()
    end
    
    local encoder = translation_model.encoder
    local decoder = translation_model.decoder
    
    local context_mat = encoder:forward(source)
    local loss = decoder:forward(in_target, context_mat, out_target)
    
    local d_context_mat = decoder:backward(in_target, context_mat, out_target)
    encoder:backward(source, d_context_mat)
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

local function main()
    local profiler = newProfiler()
    profiler:start()
    
    local train_losses = {}
    local val_losses = {}
    local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
    local iterations = opt.max_epochs * loader.ntrain
    local iterations_per_epoch = loader.ntrain
    local loss0 = nil
    for i = 1, 100 do
        local epoch = i / loader.ntrain
    
        local timer = torch.Timer()
        local _, loss = optim.rmsprop(feval, params, optim_state)
        local time = timer:time().real
    
        local train_loss = loss[1] -- the loss is inside a list, pop it
        train_losses[i] = train_loss
    
        -- exponential learning rate decay
        if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
            if epoch >= opt.learning_rate_decay_after then
                local decay_factor = opt.learning_rate_decay
                optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
                print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
            end
        end
    
--        -- every now and then or on last iteration
--        if i % opt.eval_val_every == 0 or i == iterations then
--            -- evaluate loss on validation data
--            local val_loss = eval_split(2) -- 2 = validation
--            val_losses[i] = val_loss
--    
--            local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
--            print('saving checkpoint to ' .. savefile)
--            local checkpoint = {}
--            checkpoint.protos = protos
--            checkpoint.opt = opt
--            checkpoint.train_losses = train_losses
--            checkpoint.val_loss = val_loss
--            checkpoint.val_losses = val_losses
--            checkpoint.i = i
--            checkpoint.epoch = epoch
--            checkpoint.vocab = loader.vocab_mapping
--            torch.save(savefile, checkpoint)
--        end
--    
--        if i % opt.print_every == 0 then
--            print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
--        end
--        print(i) 
        if i % 10 == 0 then
            print(i) 
            collectgarbage() 
        end
    
--        -- handle early stopping if things are going really bad
--        if loss[1] ~= loss[1] then
--            print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
--            break -- halt
--        end
--        if loss0 == nil then loss0 = loss[1] end
--        if loss[1] > loss0 * 3 then
--            print('loss is exploding, aborting.')
--            break -- halt
--        end
    end
    profiler:stop()
    local outfile = io.open( "/home/nghia/translate.prof.txt", "w+" )
    profiler:report( outfile )
    outfile:close()
end

main()
