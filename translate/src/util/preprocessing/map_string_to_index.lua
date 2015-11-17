local Vocab = require 'util.preprocessing.vocab'

local DictMapper = {}

local function strjoin(delimiter, list)
  local len = #list
  if len == 0 then 
    return "" 
  end
  local result = list[1]
  for i = 2, len do 
    result = result .. delimiter .. list[i] 
  end
  return result
end

function DictMapper.map(input_file, dict_file, output_file)
    local vocab = Vocab.load(dict_file)
    local word2index = Vocab.word2index(vocab)
    local out_stream = io.open(output_file,"w+")
    -- TODO: unknown word, end of line, start of line? maybe just unknown word
    -- end of line and start of line can be added during training
    local line_num = 0
    for line in io.lines(input_file) do
        line_num = line_num + 1
        if (line_num % 10000 == 0) then
            print(line_num)
        end
        line = line:lower()
        local indices = {}
        for word in string.gmatch(line, "%S+") do
             local index = word2index[word]
             if not index then
                 index = 0
             end
             table.insert(indices,index)
        end
        out_stream:write(strjoin(" ",indices))
        out_stream:write("\n")
    end 
end

return DictMapper