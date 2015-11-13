local FlatLinear, parent = torch.class('nn.FlatLinear', 'nn.Module')

----
-- Now I assume input is a table, including
--   - 1 vector (size k)
--   - 1 matrix (size m, l) where m is the size of each individual small attentee
--     vectors and l is the number of these vectors
function FlatLinear:__init(size)
   parent.__init(self)

   self.gradInput = torch.Tensor()
   
   self.weight = torch.Tensor(1,size)
   self.gradWeight = torch.Tensor(1,size)

   self:reset()
end

function FlatLinear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      self.weight:apply(function()
            return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
   end
   
   return self
end

function FlatLinear:updateOutput(input)
   -- TODO:
   -- understand why they don't have this guy here 
   -- if don't understand, uncomment the line below
   self.output:zero()
   
   if input:dim() == 2 then
      self.output:resize(1,input:size(2))
      self.output:mm(self.weight, input) 
      self.output:resize(input:size(2))
   elseif input:dim() == 3 then
      local batch_size = input:size(1)
      local seq_size = input:size(2)
      local output_size = input:size(3)
      input:resize(batch_size * seq_size, output_size)
      self.output:resize(batch_size * seq_size, 1)
      self.output:mm(input, self.weight:t())
      self.output:resize(batch_size, seq_size)
      input:resize(batch_size, seq_size, output_size)
   else
      error('input must be a matrix or cube')
   end
   return self.output
end



function FlatLinear:updateGradInput(input, gradOutput)
   
   if input:dim() == 2 then
      local tmp = torch.Tensor(gradOutput:size(1))
      tmp:copy(gradOutput)
      tmp:resize(1,gradOutput:size(1))
      self.gradInput:resizeAs(input)
      self.gradInput:zero()
      self.gradInput:addmm(1,self.weight:t(), tmp) -- switch if change dimension of attentee
      
   elseif input:dim() == 3 then
      local batch_size = input:size(1)
      local seq_size = input:size(2)
      local output_size = input:size(3)
      gradOutput:resize(batch_size * seq_size, 1)
      self.gradInput:resize(batch_size * seq_size, output_size)
      
      self.gradInput:mm(gradOutput, self.weight)
      self.gradInput:resize(batch_size, seq_size, output_size)
      gradOutput:resize(batch_size, seq_size, 1)
   else
      error('input must be a matrix or cube')
   end
   return self.gradInput
end


function FlatLinear:accGradParameters(input, gradOutput, scale)
   
   scale = scale or 1
   -- TODO: fix this, don't rotate input?
   if input:dim() == 2 then
      local tmp = torch.Tensor(gradOutput:size(1))
      tmp:copy(gradOutput)
      tmp:resize(1,gradOutput:size(1))
      self.gradWeight:addmm(scale, tmp, input:t())
   elseif input:dim() == 3 then
      local batch_size = input:size(1)
      local seq_size = input:size(2)
      local output_size = input:size(3)
      gradOutput:resize(batch_size * seq_size, 1)
      input:resize(batch_size * seq_size, output_size)
      
      self.gradWeight:mm(gradOutput:t(), input)
      
      input:resize(batch_size, seq_size, output_size)
      gradOutput:resize(batch_size, seq_size, 1)
   else
      error('input must be a matrix or cube')
   end
end

-- we do not need to accumulate parameters when sharing
FlatLinear.sharedAccUpdateGradParameters = FlatLinear.accUpdateGradParameters


function FlatLinear:__tostring__()
  return torch.type(self) ..
      string.format('(%d and k -> 1 x k)', self.weight:size(2))
end