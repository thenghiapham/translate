local Shuffler = {}

function Shuffler.read_lines(input_filename)
    local lines = {}
    for line in io.lines(input_filename) do
        table.insert(lines, line)
    end
    return lines
end

function Shuffler.save_lines(output_filename)
    local lines = {}
    local file = io.open(output_filename,"w+")
    for i=1,#lines do
        file:write(lines.."\n")
    end
    file:flush()
    file:close()
    return lines
end

function Shuffler.shuffle(input_filename1, input_filename2, output_filename1, 
        output_filename2)
    local input_lines1 = Shuffler.read_lines(input_filename1)
    local input_lines2 = Shuffler.read_lines(input_filename2)
    local line_num = #input_lines1
    local random = math.random
    if (line_num == #input_lines2) then
        local tmp
        local j
        for i=2,line_num do
            j = random(i)
            tmp = input_lines1[i]
            input_lines1[i] = input_lines1[j]
            input_lines1[j] = tmp
            tmp = input_lines2[i]
            input_lines2[i] = input_lines2[j]
            input_lines2[j] = tmp
        end
        Shuffler.save_lines(input_lines1, output_filename1)
        Shuffler.save_lines(input_lines2, output_filename2)
    else
        error("number of lines are different")
    end     
end