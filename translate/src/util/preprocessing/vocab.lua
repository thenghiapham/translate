require 'torch'

local Vocab = {}

local function sorted_pairs(t)

    -- collect the keys
    local keys = {}
    for k in pairs(t) do keys[#keys+1] = k end

    -- if order function given, sort by it by passing the table and keys a, b,
    -- otherwise just sort the keys 
    table.sort(keys, function(a,b) return t[b] < t[a] end)

    -- return the iterator function
    local i = 0
    return function()
        i = i + 1
        if keys[i] then
            return keys[i], t[keys[i]]
        end
    end
end



function Vocab.create_dictionary(text_file)
    local dict = {}
    for line in io.lines(text_file) do
        for word in string.gmatch(line, "%S+") do
             if (dict[word]) then
                 dict[word] = dict[word] + 1
             else
                 dict[word] = 1
             end
        end 
    end
    return dict
end


function Vocab.filter(dict, minimum_count)
    local new_dict = {}
    for k,v in pairs(dict) do
        if (v >= minimum_count) then
            new_dict[k] = v
        end
    end
    return new_dict
end

function Vocab.save(dict, filename)
    local file = io.open(filename,'w+')
    for k,v in sorted_pairs(dict) do
        file:write(k .. "\t" .. v .. "\n")    
    end
    
    file:close()
end


function Vocab.load(filename)
    local dict = {}
    for line in io.lines(filename) do
        local elements = string.gmatch(line, "%S+")
        dict[elements()] = tonumber(elements())
    end
    return dict
end

function Vocab.word2index(dict)
    local w2i = {}
    for k in sorted_pairs(dict) do
        w2i[k] = #w2i+1    
    end
    return w2i
end

function Vocab.index2word(dict) 
    local i2w = {}
    for k in sorted_pairs(dict) do
        i2w[#i2w+1] = k     
    end
    return i2w
end
--local dict = Vocab.create_dictionary("/home/nghia/test_io_lua.txt")
--local filtered_dict = Vocab.filter(dict,2)
--Vocab.save(filtered_dict, "/home/nghia/test_dict.out")

--local dict = Vocab.load("/home/nghia/test_dict.out")
--Vocab.save(dict, "/home/nghia/test_dict1.out")