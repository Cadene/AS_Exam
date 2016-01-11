require 'torch'
require 'os'

CharLMMinibatchLoader = require 'CharLMMinibatchLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Convert data to torch format')
cmd:text()
cmd:text('Options')
cmd:option('-txt','input.txt','data source')
cmd:option('-vocab','vocab.t7','name of the char->int table to save')
cmd:option('-data','data.t7','name of the serialized torch ByteTensor to save')
cmd:text()

params = cmd:parse(arg)
 
if false then
os.execute("cat " .. params.txt .. " | " ..
    -- "grep -v '[0-9]' | " ..                      -- ignore lines containing digits
    "tr '[:upper:]\n' '[:lower:] ' | " ..           -- lowercase all letters
    -- remove characters besides a-z, :;.?!(), comma, space (NOTE: WE REMOVE \n!!!)
    "tr -d -c '[:digit:][:lower:]:;.?!)(, ' | " ..  
    "tr -s ' ' > tmp.txt")                     -- squash extra spaces together
end

-- build data.t7 and vocab.t7
CharLMMinibatchLoader.text_to_tensor('tmp.txt', params.vocab, params.data)

-- os.execute("rm data/tmp.txt")
