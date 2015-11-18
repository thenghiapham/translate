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
    local line_num = 0
    for line in io.lines(text_file) do
        line_num = line_num + 1
        if (line_num % 10000 == 0) then
            print(line_num)
        end
        line = line:lower()
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

function Vocab.top(dict, n)
    local top_dict = {}
    local num = 1
    for k,v in sorted_pairs(dict) do
        top_dict[k] = v
        if (num == n) then
            break
        end
        num = num + 1
    end
    return top_dict
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
    local index = 1
    for k in sorted_pairs(dict) do
        w2i[k] = index
        index = index + 1    
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

function Vocab.size(dict)
    local num = 0
    for k in pairs(dict) do
        num = num + 1     
    end
    return num
end

function Vocab.create_mapping(dict_size, no_clusters)
    dict_size = dict_size or Vocab.size(dict)
    local mod = dict_size % no_clusters
    local cluster_size = (dict_size - mod) / no_clusters
    
    local split_point = mod * (cluster_size + 1)
    local mapping = {}
    for t = 1, split_point do
        local in_cluster_index = (t - 1) % (cluster_size + 1) + 1
        local cluster_index = (t - in_cluster_index) / (cluster_size + 1) + 1
        local indices = {cluster_index, in_cluster_index}
        table.insert(mapping,indices)
    end
    for t = split_point + 1, dict_size do
        local in_cluster_index = (t - split_point - 1) % cluster_size + 1
        local cluster_index = (t - split_point - in_cluster_index) / cluster_size + 1 + mod
        local indices = {cluster_index, in_cluster_index}
        table.insert(mapping,indices)
    end
    return mapping
end

return Vocab
