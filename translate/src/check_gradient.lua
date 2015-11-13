require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'util.table_utils'

local FakeLoader = require 'util.loader_utils'
local TranslationModel = require 'model.TranslationModel'
local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
-- model params
cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-source_vocab_size',4,'number of words in the source vocab')
cmd:option('-target_vocab_size',6,'number of words in the target vocab')
cmd:option('-rnn_size', 3, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
  -- optimization
cmd:option('-max_seq_length',10,'maximum sequence length')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size',0,'number of sequences to train on in parallel')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()
  
-- parse input params
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

local translation_model = TranslationModel.create_model(opt)
local params = translation_model.params
local grad_params = translation_model.grad_params
local loader = FakeLoader.create()

local function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

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
    
    return loss, grad_params
end

local initialization_file = "/home/nghia/test_translation.txt"
if not file_exists(initialization_file) then
    params:uniform(-0.2, 0.2)
    local param_table = {}
    for t = 1,params:size(1) do
        param_table[t] = params[t]
    end
    table.save(param_table, initialization_file)
    print("save")
else
    local param_table = table.load(initialization_file)
    local saved_params = torch.Tensor(param_table)
    
    if opt.gpuid >= 0 and opt.opencl == 0 then
        saved_params = saved_params:cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then
        saved_params = saved_params:cl()
    end
    params:copy(saved_params)
    print("load")
end

local diff,dC,dC_est = optim.checkgrad(feval, params, 1e-7)
--eval(params)
print(diff)
local merge = torch.cat({dC, dC_est},2)
print(merge)