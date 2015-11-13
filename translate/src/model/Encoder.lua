local Encoder = {}

Encoder.__index = Encoder

function Encoder.create(opt, embeddings, forward_rnns, backward_rnns, init_state)
    local self = {}
    setmetatable(self, Encoder)
    
    self.opt = opt
    self.embeddings = embeddings
    self.forward_rnns = forward_rnns
    self.backward_rnns = backward_rnns
    self.init_state = init_state
    
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
    
    local forward_states = {[0] = self.init_state}
    local backward_states = {[seq_length+1] = self.init_state}
    local predictions = {}           
    local loss = 0
--    error("Stop here")
    local embeddings = {}
    
    -- get embedding
    
    local context_lists = {{},{}}
    for t=1,seq_length do
        self.embeddings[t]:training()
        -- TODO: create a wrapper for this embedding
        embeddings[t] = self.embeddings[t]:forward(input_sequence[t])
    end
    
    
    -- forward through bidirectional lstm
    local backward_t = 0
    for t=1,seq_length do
        -- make sure we are in correct mode (this is cheap, sets flag)
        self.forward_rnns[t]:training() 
        self.backward_rnns[t]:training()
        
        local fst = self.forward_rnns[t]:forward{embeddings[t], unpack(forward_states[t-1])}
        backward_t = 1 + seq_length- t
        
        local bst = self.backward_rnns[backward_t]:forward{embeddings[backward_t], unpack(backward_states[backward_t + 1])}
        
        -- since no prediction, put it straight away
        forward_states[t] = fst
        backward_states[backward_t] = bst
        
        -- put the last hidden into context vector lists
        context_lists[1][t] = fst[#fst]
        context_lists[2][backward_t] = bst[#bst]
    end
    
    -- TODO: merge the context vector lists into a context matrix
end