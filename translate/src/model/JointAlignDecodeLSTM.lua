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
--  table.insert(outputs,context_vector)
  return nn.gModule(inputs, outputs)
end

return JointAlignDecodeLSTM

