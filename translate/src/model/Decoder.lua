local Decoder = {}

Decoder.__index = Decoder

function Decoder.create(opt, embeddings, aligned_rnns, criteria, init_state)
    local self = {}
    setmetatable(self, Decoder)
    
    self.opt = opt
    self.embeddings = embeddings
    self.aligned_rnns = aligned_rnns
    self.criteria = criteria
    self.init_state = init_state
    
    return self
end

---- Maybe the input sequence should have start_sentence, unk, end_sentence so
-- that we don't have to worry about shit

-- for now max_seq_length is for both source and target language, maybe change later
function Decoder:forward(input_sequence)
    ---- assuming input_sequence is a tensor, the first dimension is the length
    -- of the sequence
    -- return the context matrix
    local opt = self.opt
    local seq_length = input_sequence.size(1)
    local max_seq_length = opt.max_seq_length
    if (max_seq_length < seq_length) then
        error("input sequence is longer than maximum length %d", max_seq_length)
    end
    
    local states = {[0] = self.init_state}
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
    for t=1,seq_length do
        -- make sure we are in correct mode (this is cheap, sets flag)
        self.aligned_rnns[t]:training() 
        
        local fst = self.aligned_rnns[t]:forward{embeddings[t], unpack(states[t-1])}
        
        
        -- TODO: split state here because there is prediction
        states[t] = fst
        
        -- TODO: criterion, loss
    end
    
    -- TODO: return loss here
end