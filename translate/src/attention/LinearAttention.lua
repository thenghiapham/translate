local LinearAttention, parent = torch.class('nn.LinearAttention', 'nn.Module')

----
-- Now I assume input is a table, including
--   - 1 vector (size k)
--   - 1 matrix (size m, l) where m is the size of each individual small attentee
--     vectors and l is the number of these vectors
function LinearAttention:__init(factorSize, attenteeSize, output_size)
   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   
   self.weightFactor = torch.Tensor(output_size, factorSize)
   self.weightAttentee = torch.Tensor(output_size, attenteeSize)
   self.gradWeightFactor = torch.Tensor(output_size, factorSize)
   self.gradWeightAttentee = torch.Tensor(output_size, attenteeSize)
   self:reset()
end

function LinearAttention:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weightAttentee:size(2))
   end
   if nn.oldSeed then
      self.weightFactor:apply(function()
            return torch.uniform(-stdv, stdv)
      end)
      self.weightAttentee:apply(function()
            return torch.uniform(-stdv, stdv)
      end)
      
   else
      self.weightFactor:uniform(-stdv, stdv)
      self.weightAttentee:uniform(-stdv, stdv)
   end
   return self
end

function LinearAttention:parameters()
   return {self.weightFactor, self.weightAttentee}, 
          {self.gradWeightFactor, self.gradWeightAttentee}
end

function LinearAttention:updateOutput(input)
   self.output:zero()
   local factor = input[1]
   local attentee = input[2]
   
   if attentee:dim() == 2 and factor:dim() == 1 then
      local nframe = attentee:size(2)
      local nElement = self.output:nElement()
      
      self.output:resize(self.weightFactor:size(1),nframe)
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      local factorShare = torch.mv(self.weightFactor,factor)
      self.output:addmm(self.weightAttentee, attentee) 
      self.output:addr(factorShare, torch.Tensor(attentee:size(2)):fill(1)) -- add all element 
   elseif attentee:dim() == 3 and factor:dim() == 2 then
      --[[ In this case, the sequences must have the same lengths because the
        batch operation put everything into a cube
        input   attentee b x seq x rnn*k
        wfact   rnn*k x out
        output  b x seq x out 
      ]]--
      local batch_size = attentee:size(1)
      local seq_size = attentee:size(2)
      local attentee_size = attentee:size(3)
      local output_size = self.weightFactor:size(1)
      local nElement = self.output:nElement()
      
      self.output:resize(batch_size * seq_size, output_size)
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.output:addmm(attentee:resize(batch_size * seq_size, attentee_size), self.weightAttentee:t())
      local temp = torch.ger(torch.Tensor(seq_size):fill(1),(factor * self.weightFactor:t()):resize(batch_size * output_size))
      temp:resize(seq_size, batch_size, output_size)
      self.output:add(temp:transpose(1,2))
      
      attentee:resize(batch_size, seq_size, attentee_size)
      self.output:resize(batch_size, seq_size, output_size)
   else
      error('input must be a table of a vector and a matrix (single) or a matrix and a cube')
   end
   return self.output
end



function LinearAttention:updateGradInput(input, gradOutput)
   local factor = input[1]
   local attentee = input[2]
   
   local gradFactor = self.gradInput[1]
   local gradAttentee = self.gradInput[2]
   local factorElement = gradFactor:nElement()
   local attenteeElement = gradAttentee:nElement()
   
   if attentee:dim() == 2 and factor:dim() == 1 then
      gradFactor:resizeAs(factor)
      gradAttentee:resizeAs(attentee)
      gradFactor:zero()
      gradAttentee:zero()
      gradAttentee:addmm(1, self.weightAttentee:t(), gradOutput) -- switch if change dimension of attentee
      gradFactor:addmv(self.weightFactor:t(),gradOutput:sum(2):resize(gradOutput:size(1)))
   elseif attentee:dim() == 3 and factor:dim() == 2 then
      local batch_size = gradOutput:size(1)
      local seq_size = gradOutput:size(2)
      local output_size = gradOutput:size(3)
      local attentee_size = self.weightAttentee:size(2)
      local factor_size = self.weightFactor:size(2)
      
      gradAttentee:resize(batch_size * seq_size, attentee_size)
      gradFactor:resizeAs(factor)
      gradFactor:zero()
      gradAttentee:zero()
      
      
      gradFactor:addmm(gradOutput:sum(2):resize(batch_size, output_size), self.weightFactor)
      
      gradOutput:resize(batch_size * seq_size, output_size)
      gradAttentee:addmm(gradOutput, self.weightAttentee)
      
      gradAttentee:resizeAs(attentee)
      gradOutput:resize(batch_size, seq_size, output_size)
   else
      error('input must be a table of a vector and a matrix (single) or a matrix and a cube')
   end
   return self.gradInput
end


function LinearAttention:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local factor = input[1]
   local attentee = input[2]
   if attentee:dim() == 2 and factor:dim() == 1 then
      self.gradWeightAttentee:addmm(scale, gradOutput, attentee:t())
      self.gradWeightFactor:addr(scale , gradOutput:sum(2):resize(gradOutput:size(1)), factor)
   elseif  attentee:dim() == 3 and factor:dim() == 2 then
      local batch_size = gradOutput:size(1)
      local seq_size = gradOutput:size(2)
      local output_size = gradOutput:size(3)
      local attentee_size = self.weightAttentee:size(2)
      local factor_size = self.weightFactor:size(2)
      
      self.gradWeightFactor:addmm(scale, gradOutput:sum(2):resize(batch_size, output_size):t(), factor)
      
      attentee:resize(batch_size * seq_size, attentee_size)
      gradOutput:resize(batch_size * seq_size, output_size)
      
      self.gradWeightAttentee:addmm(gradOutput:t(), attentee)
      
      attentee:resize(batch_size, seq_size, attentee_size)
      gradOutput:resize(batch_size, seq_size, output_size)
   else
      error('input must be a table of a vector and a matrix (single) or a matrix and a cube')
   end
end

-- we do not need to accumulate parameters when sharing
LinearAttention.sharedAccUpdateGradParameters = LinearAttention.accUpdateGradParameters


function LinearAttention:__tostring__()
  return torch.type(self) ..
      string.format('(%d and %d -> attention)', self.weightFactor:size(1), self.weightAttentee:size(1))
end