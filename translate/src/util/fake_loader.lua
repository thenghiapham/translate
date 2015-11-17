local FakeLoader = {}
FakeLoader.__index = FakeLoader

function FakeLoader.create()
    local self = {}
    setmetatable(self, FakeLoader)
    return self
end

function FakeLoader:next_batch()
    local x = torch.Tensor{1,2,3,4,3}
    local y1 = torch.Tensor{5,3,1,4,2,4}
    local y2 = torch.Tensor{3,1,4,2,4,6}
    return x,y1,y2
end

return FakeLoader

