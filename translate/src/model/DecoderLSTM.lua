
local DecoderLSTM = {}

--
function DecoderLSTM.lstm(rnn_size, n, dropout, use_batch)
  -- TODO: try to put the embedding layers here as parameters
  -- then clone both forward and backward after flattening
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- context c
  table.insert(inputs, nn.Identity()()) -- prev_c[L]
  table.insert(inputs, nn.Identity()()) -- prev_h[L]

  local x
  local outputs = {}
  
  -- c,h from previos timesteps
  local prev_h = inputs[L*2+1]
  local prev_c = inputs[L*2]
  -- the input to this layer
    x = inputs[1]
  -- evaluate the input sums at once for efficiency
  local i2h = nn.Linear(rnn_size, 4 * rnn_size)(x)
  -- TODO: change the context size if needed
  local c2h = nn.Linear(rnn_size * 2, 4 * rnn_size)(x) -- for now say the context is bidirectional
  local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
  local all_input_sums = nn.CAddTable()({i2h, c2h, h2h})

  local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)

  -- batch -> split table (2)
  -- no batch -> split table (1)
  -- should be consistent somehow
  local n1, n2, n3, n4
  if (use_batch) then
      n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
  else
      n1, n2, n3, n4 = nn.SplitTable(1)(reshaped):split(4)
  end

  -- decode the gates
  local in_gate = nn.Sigmoid()(n1)
  local forget_gate = nn.Sigmoid()(n2)
  local out_gate = nn.Sigmoid()(n3)
  -- decode the write inputs
  local in_transform = nn.Tanh()(n4)
  -- perform the LSTM update
  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })
  -- gated cells form the output
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  
  table.insert(outputs, next_c)
  table.insert(outputs, next_h)
  
  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)
  return nn.gModule(inputs, outputs)
  
  -- TODO: put the criterion if necessary
end

return DecoderLSTM

