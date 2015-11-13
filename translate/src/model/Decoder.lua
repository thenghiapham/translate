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
    
    for t = 1,opt.max_seq_length do
        if (opt.use_batch) then
            self.d_word_vectors[t] = torch.zeros(opt.batch_size, opt.rnn_size)
        else
            self.d_word_vectors[t] = torch.zeros(opt.rnn_size)
        end
    end
    
    self.d_context_matrix = torch.Tensor()
    return self
end

---- Maybe the input sequence should have start_sentence, unk, end_sentence so
-- that we don't have to worry about shit

-- for now max_seq_length is for both source and target language, maybe change later
-- input sequence has ^, start_of_line
-- output sequence has $, end_of_line
function Decoder:forward(input_sequence, context_matrix, output_sequence)
    ---- assuming input_sequence and output_sequence are more or less the same
    -- but shifted by one word
    local opt = self.opt
    local seq_length = input_sequence.size(1)
    local max_seq_length = opt.max_seq_length
    if (max_seq_length < seq_length) then
        error("input sequence is longer than maximum length %d", max_seq_length)
    end
    
    self.states = {[0] = self.init_state}
    local predictions = {}           
    local loss = 0
    
    self.word_vectors = {}
    -- get embedding
    for t=1,seq_length do
        self.embeddings[t]:training()
        -- TODO: create a wrapper for this embedding
        self.word_vectors[t] = self.embeddings[t]:forward(input_sequence[t])
    end
    
        
    self.predictions = {}           -- softmax outputs
    local loss = 0
    -- forward through bidirectional lstm
    for t=1,seq_length do
        -- make sure we are in correct mode (this is cheap, sets flag)
        self.aligned_rnns[t]:training() 
        local rst = self.aligned_rnns[t]:forward{self.word_vectors[t], context_matrix, unpack(self.states[t-1])}
        
        for i=1,#self.init_state do
            table.insert(self.states[t], rst[i])
        end -- extract the state, without output
        
        predictions[t] = rst[#rst]
        loss = loss + self.criteria[t]:forward(predictions[t], output_sequence[t])
    end
    
    -- TODO: when do gradient checking, can't average (don-t want to divide gradParmams)
--    loss = loss / opt.seq_length
    return loss
    
end

function Decoder:backward(input_sequence, context_matrix, output_sequence)
    local opt = self.opt
    local seq_length = #input_sequence
     
    local d_states = {[seq_length] = clone_list(self.init_state, true)} -- true also zeros the clones
    
    for t=1,seq_length do
        self.d_word_vectors[t]:zero()
    end
    
    self.d_context_matrix.resizeAs(context_matrix)
    self.d_context_matrix:zero()
    
    for t=seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = self.criteria[t]:backward(self.predictions[t], y[{{}, t}])
        table.insert(d_states[t], doutput_t)
        local drst = self.aligned_rnns[t]:backward({self.word_vectors[t], unpack(self.states[t-1])}, d_states[t])
        
        d_states[t-1] = {}
        -- the loop is not necessrary but leave it for now
        for k,v in pairs(drst) do
            if k > 2 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                d_states[t-1][k-2] = v
            end
        end
        self.d_word_vectors[t]:add(drst[1])
        self.d_context_matrix:add(drst[2])
    end
    return self.d_context_matrix
end