require 'torch'
local tensor_utils = {}

---- use to merge the list of (bidirectional) lstm final layer's vectors into a matrix
function tensor_utils.merge(list_of_vector_list)
    
    local vector_dimension = list_of_vector_list[1][1]:dim()
    local seq_size = #list_of_vector_list
    
    if vector_dimension == 1 then
        -- make it seq x rnn like the 3D case?
        local list_of_mat = {}
        for t = 1,seq_size do
            list_of_mat[t] = torch.cat(list_of_vector_list[t],2)
        end
        local result = torch.cat(list_of_mat, 1)
        return result
    else
        local new_list = {}
        local first_vector = list_of_vector_list[1][1]
        local batch_size = first_vector:size(1)
        
        local vector_length = first_vector:size(2)
        local new_vector_length = vector_length * seq_size
        local element_list_length = #(list_of_vector_list[1])
        
        for t = 1,(seq_size * element_list_length) do
            local list_index = (t % seq_size) + 1
            new_list[t] = list_of_vector_list[list_index][(t - list_index + 1) / seq_size]
        end
        
        local result = torch.cat(new_list,2)
        return result:resize(batch_size, seq_size, new_vector_length)
    end
end

---- inverse of the above function
function tensor_utils.cut_vectors(matrix, num_split)
    local list_of_vector_list = {}
    for t = 1,num_split do
        list_of_vector_list[t] = {}
    end
    
    if matrix:dim() == 2 then 
        local matrix_t = matrix:t()
        
        local vector_length = matrix:size(1)
        local new_vector_length = vector_length / num_split
        for i = 1,matrix:size(2) do
            for t = 1,num_split do
                local vector = torch.Tensor(new_vector_length)
                vector:copy(matrix:sub((t-1) * new_vector_length + 1, t * new_vector_length,i,i))
                table.insert(list_of_vector_list[t], vector)
            end
        end
    else
        local batch_size = matrix:size(1)
        local seq_size = matrix:size(2)
        local length = matrix:size(3)
        local new_vector_length = vector_length / num_split
        for i = 1,seq_size do
            for t = 1,num_split do
                local vector = torch.Tensor(batch_size, new_vector_length)
                vector:copy(1,batch_size,i,i,matrix:sub((t-1) * new_vector_length + 1, t * new_vector_length))
                table.insert(list_of_vector_list[t], vector)
            end
        end
        
    end
    return list_of_vector_list
end


function tensor_utils.extract_last_index(mat, i,j)
    if mat:dim() == 1 then
        return mat:sub(i,j)
    elseif mat:dim() == 2 then
        local fist_size = mat:size(1)
        local result = mat:sub(1,fist_size, i, j)
        return result
    elseif mat:dim() == 3 then
        local fist_size = mat:size(1)
        local second_size = mat:size(2)
        return mat:sub(1,fist_size, 2, second_size, i, j)
    else
        error("Don't want to deal with this")
    end
end

return tensor_utils