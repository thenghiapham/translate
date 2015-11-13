
local Alignment = {}
require 'attention.LinearAttention'
require 'attention.MVMul'
require 'attention.FlatLinear'
--
function Alignment.align_model(state_size, context_size, tmp_output_size, dropout)
  -- TODO: try to put the embedding layers here as parameters
  -- then clone both forward and backward after flattening
  dropout = dropout or 0 
  local inputs={}
  local outputs={}
  
  table.insert(inputs, nn.Identity()()) -- s_(t-1)
  table.insert(inputs, nn.Identity()()) -- context matrix
  
  local prev_state = inputs[1]
  local context_mat = inputs[2]
  
  -- M1 = tanh(W1 state + (W2 s) \outer (1))
  local linearAttention = nn.LinearAttention(state_size, context_size, tmp_output_size)({prev_state, context_mat})
  local tanhAttention = nn.Tanh()(linearAttention)
  -- a =softmax(w * M1) 
  local flatAttention = nn.FlatLinear(tmp_output_size)(tanhAttention)
  local attentionWeight = nn.SoftMax()(flatAttention)
  
  --- c_v = M1 * a
  local context_vector = nn.MVMul()({context_mat, attentionWeight})

  table.insert(outputs, context_vector)
  return nn.gModule(inputs, outputs)
end

return Alignment

