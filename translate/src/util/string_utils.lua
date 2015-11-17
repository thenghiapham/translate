function strjoin(delimiter, list)
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

function split(sentence)
    local array = {}
    for word in string.gmatch(sentence, "%S+") do
         table.insert(array, word)
    end 
    return array
end

function text2num_array(sentence)
    local array = {}
    for word in string.gmatch(sentence, "%S+") do
         table.insert(array, tonumber(word))
    end 
    return array
end