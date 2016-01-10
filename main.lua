
require 'torch'
require 'nn'
require 'nngraph'

local model_utils = require 'model_utils'

local CharLMMinibatchLoader=require 'CharLMMinibatchLoader'

-- Modules --------------------------------------------------------

function module_h(dimx, dimh)
   local h_linH = nn.Linear(dimh, dimh)()
   local h_linW = nn.Linear(dimx, dimh)()
   local h_sum = nn.CAddTable()({h_linH, h_linW})
   local h_tanh = nn.Tanh()(h_sum)
   local h = nn.gModule({h_linH, h_linW}, {h_tanh})
   h.dimX = dimx
   h.dimH = dimh
   return h
end

function module_g(dimx, dimh)
   local g = nn.Sequential()
   g:add(nn.Linear(dimh,dimx))
   --g:add(nn.SoftMax())
   g:add(nn.LogSoftMax())
   g.dimX = dimx
   g.dimG = dimh
   return g
end

function module_predictNext(g, h, seqSize)
   local listH = model_utils.clone_many_times(h, seqSize)
   local listG = model_utils.clone_many_times(g, seqSize)
   -- create inputs nodes --
   local inputs = {}
   for i=1,seqSize+1 do
      inputs[i] = nn.Identity()()
   end
   -- create H nodes --
   local listGraphH = {}
   listGraphH[1] = listH[1]({inputs[1], inputs[2]})
   for i=2,seqSize do
      listGraphH[i] = listH[i]({listGraphH[i-1], inputs[i+1]})
   end
   -- create G nodes --
   local listGraphG = {}
   for i=1,seqSize do
      listGraphG[i] = listG[i]({listGraphH[i]})
   end
   -- create complete graph --
   local rnn = nn.gModule(inputs, listGraphG)
   rnn.module_g = g
   rnn.module_h = h
   rnn.dimH = h.dimH
   rnn.dimX = h.dimX
   graph.dot(rnn.fg, 'RNN', 'RNN')
   return rnn
end


---------------------------------------------------------------------

-- 4 ------------------------------------------------
function prepareDataset(dataset)
   local x = {}
   --for i=1,dataset.batch_size do
   for i=1,dataset.nbatches do      
      x[i] = {} -- 
      for j=1,dataset.seq_length do
	 x[i][j] = torch.zeros(dataset.vocab_size)
	 x[i][j][dataset.x_batches[i][1][j]] = 1
      end
   end
   dataset.x = x
end

function parallelNLLCriterion(nCriterion)
   criterion = nn.ParallelCriterion()
   for i=1,nCriterion do
      criterion:add(nn.ClassNLLCriterion())
   end
   return criterion
end

function train(dataset, model, criterion, learningRate, nbIter)
   local learningRate = learningRate or 1e-2
   local nbIter = nbIter or 10
   local h0 = torch.zeros(model.dimH)
   for iter = 1,nbIter do
      local loss = 0
      local permutation = torch.randperm(dataset.nbatches)
      for p=1,dataset.nbatches do
	 local i = permutation[p]
	 local input = {}
	 input[1] = h0
	 for k=1,dataset.seq_length do
	    input[k+1] = dataset.x[i][k]
	 end
	 model:zeroGradParameters()
	 local out = model:forward(input)
	 local err = criterion:forward(out, dataset.y_batches[i][1])
	 local delta = criterion:backward(out, dataset.y_batches[i][1])
	 model:backward(input, delta)
	 model:updateParameters(learningRate)
	 loss = loss + err
      end
      print(iter, loss / dataset.nbatches)
   end
end

function generate(model, vocab, initSeq, nbIter, temperature)
   local temperature = temperature or 1.
   local h0 = torch.zeros(model.dimH)
   local seqSize = #initSeq
   local currSeq = initSeq
   local input = {}
   local output = ""
   input[1] = h0
   for k=1,seqSize do
      input[k+1] = initSeq[k]
   end   
   for inter = 1,nbIter do   
      local nextDistribution = torch.exp(torch.div(model:forward(input)[seqSize], temperature))
      nextDistribution:div(torch.sum(nextDistribution))
      local pred = torch.multinomial(nextDistribution:float(), 1):resize(1):float()[1]
      output = output .. vocab[pred]
      for k=2,seqSize do
	 input[k] = input[k+1]
      end
      nextDistribution:fill(0)
      nextDistribution[pred] = 1
      input[seqSize + 1] = nextDistribution
   end
   return output
end

function buildReverseMapping(v)
   local reverseMapping = {}
   for char, index in pairs(v.vocab_mapping) do
      reverseMapping[index] = char
   end
   v.reverseMapping = reverseMapping
end

local seqSize = 30
local v=CharLMMinibatchLoader.create("data.t7","vocab.t7",1,seqSize)
local dimx = v.vocab_size
local dimh = 200
buildReverseMapping(v)

-- module --
local h = module_h(dimx, dimh)
local g = module_g(dimx, dimh)
rnn = module_predictNext(g, h, seqSize)

criterion = parallelNLLCriterion(seqSize)

-- train --
prepareDataset(v)

train(v, rnn, criterion, 1e-5, 500)

-- generate --
local initSeq = v.x[1]
for i = 1,10 do
   local output = generate(rnn, v.reverseMapping, initSeq, 500)
   print(output)
end
dataset = v

torch.save('/Vrac/3000693/AS_Exam/model.t7', rnn)
