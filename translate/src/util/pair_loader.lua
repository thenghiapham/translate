local PairLoader = {}
PairLoader.__index = PairLoader

require 'util.string_utils'
require 'torch'

function PairLoader.create(source_file, target_file, target_dict_size)
    local self = {}
    setmetatable(self, PairLoader)
    local source_stream = io.open(source_file,"r")
    local target_stream = io.open(source_file,"r")
    self.source_stream = source_stream
    self.target_stream = target_stream
    self.start_end_word = target_dict_size + 1
    return self
end

function PairLoader:close()
    self.source_stream:close()
    self.target_stream:close()
end

function PairLoader:next_batch()
    local source_sentence = self.source_stream:read()
    if source_sentence == nil then
        return nil
    end
    local target_sentence = self.target_stream:read()
    local source_array = text2num_array(source_sentence)
    local target_array = text2num_array(target_sentence)
    -- in_target is the input of the decoder
    -- target_array (no need to create out_target) with is the output of the decoder
    -- self.start_end_word is both the start_string of in_target and end_string of out_target
    local in_target = {self.start_end_word}
    for i = 1,#target_array do
        table.insert(in_target, target_array[i])
    end
    table.insert(target_array, self.start_end_word)
    return torch.Tensor(source_array), torch.Tensor(in_target), torch.Tensor(target_array)
end

return PairLoader