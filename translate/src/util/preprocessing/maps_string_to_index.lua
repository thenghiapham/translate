local Vocab = require 'util.preprocessing.vocab'

local DictMapper = {}

local function strjoin(delimiter, list)
  local len = getn(list)
  if len == 0 then 
    return "" 
  end
  local string = list[1]
  for i = 2, len do 
    string = string .. delimiter .. list[i] 
  end
  return string
end

function DictMapper(input_file, dict_file, output_file)
    local vocab = Vocab.load(dict_file)
    local word2index = Vocab.word2index(vocab)
    local out_stream = io.open(output_file,"w+")
    -- TODO: unknown word, end of line, start of line? maybe just unknown word
    -- end of line and start of line can be added during training
    
    for line in io.lines(input_file) do
        local indices = {}
        for word in string.gmatch(line, "%S+") do
             table.insert(indices, word2index[word])
        end
        out_stream.write(strjoin(" ",indices))
        out_stream.write("\n")
    end 

end