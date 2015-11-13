local EncoderLSTM = require 'model.EncoderLSTM'
local JointAlignDecodeLSTM = require 'model.JointAlignDecodeLSTM'
local Embedding = require 'model.Embedding'

local Encoder = require 'model.Encoder'
local Decoder = require 'model.Decoder'

local model_utils = require 'util.model_utils'

local TranslationModel = {}
TranslationModel.initialized = false

function TranslationModel.create_init_states(opt)
    local h_source_init_state = {}
    for L=1,opt.num_layers do
        local h_init
        if (opt.use_batch) then
            h_init = torch.zeros(opt.batch_size, opt.rnn_size)
        else
            h_init = torch.zeros(opt.rnn_size)
        end
        if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
        if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
        table.insert(h_source_init_state, h_init:clone())
        table.insert(h_source_init_state, h_init:clone())
    end
    
    local h_target_init_state = {}
    local h_init
    if (opt.use_batch) then
        h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    else
        h_init = torch.zeros(opt.rnn_size)
    end
    if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(h_target_init_state, h_init:clone())
    table.insert(h_target_init_state, h_init:clone())
    
    return h_source_init_state, h_target_init_state
end

function TranslationModel.create_model(opt)
    local translation_model = {}
    translation_model.opt = opt
    if (opt.batch_size >= 1) then
        opt.use_batch = true
    else
        opt.use_batch = false
    end
    
    local prototype = {}
    prototype.source_embedding_layer = nn.LookupTable(opt.source_vocab_size, opt.rnn_size)
    prototype.encoder_forward_rnn = EncoderLSTM.lstm(opt.rnn_size, opt.num_layers, opt.dropout, opt.use_batch)
    prototype.encoder_backward_rnn = EncoderLSTM.lstm(opt.rnn_size, opt.num_layers, opt.dropout, opt.use_batch)
    
    prototype.target_embedding_layer = nn.LookupTable(opt.target_vocab_size, opt.rnn_size)
    prototype.decoder_align_rnn = JointAlignDecodeLSTM.lstm(opt.rnn_size, opt.rnn_size * 2, opt.target_vocab_size, opt.dropout, opt.use_batch)
    prototype.criterion = nn.ClassNLLCriterion()
    
    if opt.gpuid >= 0 and opt.opencl == 0 then
        for k,v in pairs(prototype) do
            v:cuda()
        end
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then
        for k,v in pairs(prototype) do v:cl() end
    end
    
    local params, grad_params = model_utils.combine_all_parameters(prototype.source_embedding_layer, 
            prototype.encoder_forward_rnn, prototype.encoder_backward_rnn, 
            prototype.target_embedding_layer, prototype.decoder_align_rnn)
            
    translation_model.params = params
    translation_model.grad_params = grad_params
    
    local source_embedding_clones = model_utils.clone_many_times(prototype.source_embedding_layer, opt.max_seq_length)
    local source_forward_clones = model_utils.clone_many_times(prototype.encoder_forward_rnn, opt.max_seq_length)
    local source_backward_clones = model_utils.clone_many_times(prototype.encoder_backward_rnn, opt.max_seq_length)
    
    local target_embedding_clones = model_utils.clone_many_times(prototype.target_embedding_layer, opt.max_seq_length)
    local target_align_clones = model_utils.clone_many_times(prototype.decoder_align_rnn, opt.max_seq_length)
    local criteria = model_utils.clone_many_times(prototype.criterion, opt.max_seq_length)
    
    
    local source_embeddings = {}
    local target_embeddings = {}
    for t = 1, opt.max_seq_length do
        source_embeddings[t] = Embedding.create(source_embedding_clones[t])
        target_embeddings[t] = Embedding.create(target_embedding_clones[t])
    end
    
    local h_source_init_state, h_target_init_state =  TranslationModel.create_init_states(opt)
    translation_model.encoder = Encoder.create(opt, source_embeddings, source_forward_clones, source_backward_clones, h_source_init_state)
    translation_model.decoder = Decoder.create(opt, target_embeddings, target_align_clones, criteria, h_target_init_state)
    return translation_model
end

return TranslationModel