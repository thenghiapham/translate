local DecoderLSTM = require 'model.DecoderLSTM'
local Alignment = require 'model.Alignment'

local JointAlignDecodeLSTM = {}

---- TODO:
-- 1. if add the criterion in Decoder than change in this as well
function JointAlignDecodeLSTM.lstm(rnn_size, context_size, target_vocab_size, dropout, use_batch)
  -- TODO: try to put the embedding layers here as parameters
  -- then clone both forward and backward after flattening
  dropout = dropout or 0 
  local inputs={}
  local outputs={}
  
  table.insert(inputs, nn.Identity()()) -- y (first y = $start_sequence) 
  -- TODO: decide whether to train the start sequence and end sequence
  table.insert(inputs, nn.Identity()()) -- context matrix
  table.insert(inputs, nn.Identity()()) -- c_(t-1) (first s = 0)
  table.insert(inputs, nn.Identity()()) -- s_(t-1), i.e. h (first s = 0)
  
  local prev_y = inputs[1]
  local context_mat = inputs[2]
  local prev_c = inputs[3]
  local prev_s = inputs[4]
  
  local context_vector = Alignment.align_model(rnn_size, context_size, rnn_size, dropout)({prev_s, context_mat})
  local cur_c, cur_s, pred_y = DecoderLSTM.lstm(rnn_size, target_vocab_size, dropout, use_batch)({prev_y, context_vector, prev_c, prev_s})
  table.insert(outputs, cur_c)
  table.insert(outputs, cur_s)
  table.insert(outputs, pred_y)
 
  return nn.gModule(inputs, outputs)
end

return JointAlignDecodeLSTM






--  ---- aligning
--  
--  -- M1 = tanh(W1 state + (W2 s) \outer (1))
--  local linearAttention = nn.LinearAttention(rnn_size, context_size, rnn_size)({prev_s, context_mat})
--  local tanhAttention = nn.Tanh()(linearAttention)
--  -- a =softmax(w * M1) 
--  local flatAttention = nn.FlatLinear(rnn_size)(tanhAttention)
--  local attentionWeight = nn.SoftMax()(flatAttention)
--  --- c_v = M1 * a
--  local context_vector = nn.MVMul()({context_mat, attentionWeight})
--  
--  
--  ---- translating
--  local i2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_y)
--  -- TODO: change the context size if needed
--  local context2h = nn.Linear(rnn_size * 2, 4 * rnn_size)(context_vector) -- for now say the context is bidirectional
--  local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_s)
--  local all_input_sums = nn.CAddTable()({i2h, context2h, h2h})
--
--  local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
--
--  -- batch -> split table (2)
--  -- no batch -> split table (1)
--  local n1, n2, n3, n4
--  if (use_batch) then
--      n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
--  else
--      n1, n2, n3, n4 = nn.SplitTable(1)(reshaped):split(4)
--  end
--
--  -- decode the gates
--  local in_gate = nn.Sigmoid()(n1)
--  local forget_gate = nn.Sigmoid()(n2)
--  local out_gate = nn.Sigmoid()(n3)
--  -- decode the write inputs
--  local in_transform = nn.Tanh()(n4)
--  -- perform the LSTM update
--  local next_c           = nn.CAddTable()({
--      nn.CMulTable()({forget_gate, prev_c}),
--      nn.CMulTable()({in_gate,     in_transform})
--    })
--  -- gated cells form the output
--  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
--  table.insert(outputs, next_c)
--  table.insert(outputs, next_h)
--    
--  -- set up the decoder
--  local top_h = outputs[#outputs]
--  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
--  local proj = nn.Linear(rnn_size, target_vocab_size)(top_h)
--  local logsoft = nn.LogSoftMax()(proj)
-- 
--
--  table.insert(outputs, logsoft)