
local MVMul, parent = torch.class('nn.MVMul', 'nn.Module')

function MVMul:__init()
   parent.__init(self)
   self.gradInput = {}
end

function MVMul:updateOutput(input)
   local large = input[1]
   local small = input[2]
--   print(large)
--   print(small)
   if large:dim() == 2 then
      -- TODO: current output of the whole thing is
      -- rnn x seq_length
      -- should make it seq_length x rnn like in the 3D case
      self.output:resize(large:size(1))
      self.output:mv(large,small)
   elseif large:dim() == 3 then
      local batch_size = small:size(1)
      local seq_size = small:size(2)
      local attentee_size = large:size(3)
      small:resize(batch_size, 1, seq_size)
      self.output:resize(batch_size, 1, attentee_size)
      self.output:bmm(small, large)
      small:resize(batch_size, seq_size)
      self.output:resize(batch_size, attentee_size)
--      print(self.output)
      
   else
      error('input must be a table of a vector and a matrix (single) or a matrix and a cube')
   end
   return self.output
end

-- I guess this thing has divided by zero problem
--[[function MVMul:updateGradInput_efficient(input, gradOutput)
   ... copied from CMulTable
end--]] 

function MVMul:updateGradInput(input, gradOutput)
   local large = input[1]
   local small = input[2]
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[2] = self.gradInput[2] or input[2].new()
   if large:dim() == 2 then
      
      self.gradInput[1]:resizeAs(input[1])
      self.gradInput[2]:resizeAs(input[2])
     
      self.gradInput[1]:ger(gradOutput, input[2])
      self.gradInput[2]:mv(input[1]:t(),gradOutput)
   elseif large:dim() == 3 then
      ---- gradOutput: batch x rnn/att
      --   large     : batch x seq x rnn
      --   small     : batch x seq
      --
      local batch_size = small:size(1)
      local seq_size = small:size(2)
      local attentee_size = large:size(3)
      
      gradOutput:resize(batch_size, 1, attentee_size)

      small:resize(batch_size, seq_size, 1)
      self.gradInput[1]:resizeAs(large)
      self.gradInput[1]:bmm(small,gradOutput)
      small:resize(batch_size, seq_size)
      
      gradOutput:resize(batch_size, attentee_size, 1)

      self.gradInput[2]:resize(batch_size, seq_size, 1)
      self.gradInput[2]:bmm(large, gradOutput)
      self.gradInput[2]:resizeAs(small)
      
      gradOutput:resize(batch_size, attentee_size)
   else
      error('input must be a table of a vector and a matrix (single) or a matrix and a cube')
   end
   return self.gradInput
end

function MVMul:__tostring__()
  return torch.type(self) ..
      string.format('(m x n , n --> m)')
end