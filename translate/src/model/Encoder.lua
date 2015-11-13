local Encoder = {}
Encoder.__index = Encoder

local tensor_utils = require 'util.tensor_utils' 

function Encoder.create(opt, embeddings, forward_rnns, backward_rnns, init_state)
    local self = {}
    setmetatable(self, Encoder)
    
    self.opt = opt
    self.embeddings = embeddings
    self.forward_rnns = forward_rnns
    self.backward_rnns = backward_rnns
    self.init_state = init_state
    
    self.d_word_vectors = {}
    for t = 1,opt.max_seq_length do
        if (opt.use_batch) then
            self.d_word_vectors[t] = torch.zeros(opt.batch_size, opt.rnn_size)
        else
            self.d_word_vectors[t] = torch.zeros(opt.rnn_size)
        end
    end
    
    return self
end

function Encoder:forward(input_sequence)
    ---- assuming input_sequence is a tensor, the first dimension is the length
    -- of the sequence
    -- return the context matrix
    local opt = self.opt
    local seq_length = input_sequence.size(1)
    local max_seq_length = opt.max_seq_length
    if (max_seq_length < seq_length) then
        error("input sequence is longer than maximum length %d", max_seq_length)
    end
    
    -- need to store these for backward 
    self.word_vectors = {}
    self.forward_states = {[0] = self.init_state}
    self.backward_states = {[seq_length+1] = self.init_state}
    
    local predictions = {}           
    local loss = 0
--    error("Stop here")
    
    
    -- get embedding
    
    local context_lists = {{},{}}
    for t=1,seq_length do
        self.embeddings[t]:training()
        -- TODO: create a wrapper for this embedding
        self.word_vectors[t] = self.embeddings[t]:forward(input_sequence[t])
    end
    
    
    -- forward through bidirectional lstm
    local backward_t = 0
    for t=1,seq_length do
        -- make sure we are in correct mode (this is cheap, sets flag)
        self.forward_rnns[t]:training() 
        self.backward_rnns[t]:training()
        
        local fst = self.forward_rnns[t]:forward{self.word_vectors[t], unpack(self.forward_states[t-1])}
        backward_t = 1 + seq_length- t
        
        local bst = self.backward_rnns[backward_t]:forward{self.word_vectors[backward_t], unpack(self.backward_states[backward_t + 1])}
        
        -- since no prediction, put it straight away
        self.forward_states[t] = fst
        self.backward_states[backward_t] = bst
        
        -- put the last hidden into context vector lists
        context_lists[1][t] = fst[#fst]
        context_lists[2][backward_t] = bst[#bst]
    end
    
    -- return the context matrix
    return tensor_utils.merge(context_lists)
end

function Encoder:backward(input_sequence, d_merge_state)
    -- TODO: cut
    
    local opt = self.opt
    local seq_length = input_sequence.size(1)
    local max_seq_length = opt.max_seq_length
    local num_layers = opt.num_layers
    if (max_seq_length < seq_length) then
        error("input sequence is longer than maximum length %d", max_seq_length)
    end
    
    for t=1,seq_length do
        self.d_word_vectors[t]:zero()
    end
    -- TODO: cut here
    local d_states = tensor_utils.cut_vectors(d_merge_state, 2)
    local d_forward_states = d_states[1]
    local d_backward_states = d_states[2]
    
    for t=seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        
        d_forward_states[t][2 * num_layers]:add(d_forward_states[t])
        local dfst = self.forward_rnns[t]:backward({self.word_vectors[t], unpack(self.forward_states[t-1])}, d_forward_states[t])
        
        local backward_t = 1 + seq_length - t
        d_backward_states[backward_t][2 * num_layers]:add(d_backward_states[backward_t])
        local dbst = self.backward_rnns[backward_t]:backward({self.word_vectors[backward_t], unpack(self.backward_states[backward_t+1])}, d_backward_states[backward_t])
        -- k = 1, x
        -- k = 2i: dc[i]
        -- k = 2i + 1: dh[i]
        -- k = 2 * layer + 1: need to add from attention in the next iteration
        d_forward_states[t-1] = {}
        for k,v in pairs(dfst) do
            if k > 1 then 
                d_forward_states[t-1][k-1] = v
            else 
                self.d_word_vectors[t]:add(v)
            end
        end
        
        d_backward_states[backward_t+1] = {}
        for k,v in pairs(dbst) do
            if k > 1 then 
                d_backward_states[backward_t+1][k-1] = v
            else
                self.d_word_vectors[backward_t]:add(v)
            end
        end        
    end
    
    for t=1,seq_length do
        self.embeddings[t]:backward(input_sequence[t], self.d_word_vectors[t])
    end
    

end