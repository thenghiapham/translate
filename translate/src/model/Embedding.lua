local Embedding = {}
Embedding.__index = Embedding


function Embedding.create(lookup_table)
    local self = {}
    setmetatable(self, Embedding)
    self.lookup_table = lookup_table

end

function Embedding:tensorize(index)
    if (type(index) == "number") then
        return torch.Tensor{index}
    else
        return index
    end
end
 
function Embedding:forward(input)
    if (type(input) == "number") then
        input = torch.Tensor{input}
        local word_vector = self.lookup_table:forward(input)
        return word_vector:resize(word_vector:size(2))
    else
        return self.lookup_table:forward(input)
    end
end

function Embedding:backward(input, doutput)
    if (type(input) == "number") then
        input = torch.Tensor{input}
    end
    self.lookup_table:backward(input, doutput)
end