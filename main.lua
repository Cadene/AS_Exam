
require 'torch'
require 'nn'
require 'nngraph'

local model_utils = require 'model_utils'

local CharLMMinibatchLoader=require 'CharLMMinibatchLoader'

-- RNN ---------------------------------------------------------------

function module_h(dimx, dimh)
   local h_linH = nn.Linear(dimh, dimh)()
   local h_linW = nn.Linear(dimx, dimh)()
   local h_sum = nn.CAddTable()({h_linH, h_linW})
   local h_tanh = nn.Tanh()(h_sum)
   local h = nn.gModule({h_linH, h_linW}, {h_tanh})
   h.dimx = dimx
   h.dimh = dimh
   return h
end

function module_g(dimx, dimh)
   local g = nn.Sequential()
   g:add(nn.Linear(dimh,dimx))
   --g:add(nn.SoftMax())
   g:add(nn.LogSoftMax())
   g.dimx = dimx
   g.dimg = dimh
   return g
end

function create_rnn(dimx, dimh, seqSize)
   local h = module_h(dimx, dimh)
   local g = module_g(dimx, dimh)
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
   local model = nn.gModule(inputs, listGraphG)
   model.name = 'rnn'
   model.module_g = g
   model.module_h = h
   model.dimh = h.dimh
   model.dimx = h.dimx
   model.nbhidden = 1
   graph.dot(model.fg, 'RNN', 'RNN')
   return model
end

-- LSTM -------------------------------------------------------------

function module_h_c(dimx, dimh)
   local x = nn.Identity()()
   local h = nn.Identity()()
   local c = nn.Identity()()
   local gate_i = nn.Sigmoid()(nn.CAddTable(){
      nn.Linear(dimx, dimh)(x),
      nn.Linear(dimh, dimh)(h),
      nn.Linear(dimh, dimh)(c)
   })
   local gate_f = nn.Sigmoid()(nn.CAddTable(){
      nn.Linear(dimx, dimh)(x),
      nn.Linear(dimh, dimh)(h),
      nn.Linear(dimh, dimh)(c)
   })
   local learning = nn.Tanh()(nn.CAddTable(){
      nn.Linear(dimx, dimh)(x),
      nn.Linear(dimh, dimh)(h)
   })
   local c_out = nn.CAddTable(){
      nn.CMulTable(){gate_f, c},
      nn.CMulTable(){gate_i, learning}        
   }
   local gate_o = nn.Sigmoid()(nn.CAddTable(){
      nn.Linear(dimx, dimh)(x),
      nn.Linear(dimh, dimh)(h),
      nn.Linear(dimh, dimh)(c_out)
   })
   local h_out = nn.CMulTable(){gate_o, nn.Tanh()(c_out)}
   return nn.gModule({h, c, x}, {h_out, c_out})
end

function create_lstm(dimx, dimh, seqSize)
   local nbhidden = 2
   local h_c = module_h_c(dimx, dimh)
   local g   = module_g(dimx, dimh)
   local listH = model_utils.clone_many_times(h_c, seqSize)
   local listG = model_utils.clone_many_times(g,   seqSize)
   -- create inputs nodes --
   local inputs = {}
   for i=1,seqSize+nbhidden do
      inputs[i] = nn.Identity()()
   end
   -- create LSTM nodes --
   local listGraphHC = {}
   listGraphHC[1] = listH[1]{
      inputs[1], -- hinit
      inputs[2], -- cinit
      inputs[3]  -- x1
   }
   for i=2,seqSize do
      listGraphHC[i] = listH[i]{
         nn.SelectTable(1)(listGraphHC[i-1]), -- hi
         nn.SelectTable(2)(listGraphHC[i-1]), -- ci
         inputs[i+nbhidden]   -- xi+1
      }
   end
   -- create G nodes --
   local listGraphG = {}
   for i=1,seqSize do
      listGraphG[i] = listG[i]{
         nn.SelectTable(1)(listGraphHC[i]) -- hi
      }
   end
   -- create complete graph --
   local model = nn.gModule(inputs, listGraphG)
   model.name = 'lstm'
   model.module_g = g
   model.module_h = h_c
   model.dimh = dimh
   model.dimx = dimx
   model.nbhidden = nbhidden
   graph.dot(model.fg, 'LSTM', 'LSTM')
   return model
end

---------------------------------------------------------------------

function init_h(batch_size, dimh)
  -- return torch.zeros(batch_size, dimh)
  return torch.zeros(dimh)
end
function init_c(batch_size, dimh)
  -- return torch.zeros(batch_size, dimh)
end
function onehot_encoding(vocab_size, id)
   local vector = torch.zeros(vocab_size)
   vector[id] = 1
   return vector
end

function prepareDataset(dataset, model)
   local inputs = {}
   local vocab_size = dataset.vocab_size
   local batch_size = dataset.nbatches
   for i=1,dataset.nbatches do      
      inputs[i] = {}
      if model.name == 'rnn' then
         table.insert(inputs[i], init_h(1, model.dimh))
      elseif model.name == 'lstm' then
         table.insert(inputs[i], init_h(1, model.dimh))
         table.insert(inputs[i], init_h(1, model.dimh))
      end
      for j=1,dataset.seq_length do
         local id = dataset.x_batches[i][1][j]
         table.insert(inputs[i], onehot_encoding(vocab_size, id))
      end
   end
   dataset.inputs = inputs
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
   local learningDecay = false
   local nbIter = nbIter or 10
   for iter = 1,nbIter do
      local loss = 0
      local permutation = torch.randperm(dataset.nbatches)
      for p=1,dataset.nbatches do
         local i = permutation[p]
         local inputs = dataset.inputs[i]
	      model:zeroGradParameters()
	      local out = model:forward(inputs)
	      local err = criterion:forward(out, dataset.y_batches[i][1])
	      local delta = criterion:backward(out, dataset.y_batches[i][1])
	      model:backward(inputs, delta)
	      model:updateParameters(learningRate)
	      loss = loss + err
      end
      if learningDecay then
         if iter > 40 then
            learningRate = learningRate / 2
         end
      end
      print(iter, loss / dataset.nbatches)
   end
end

function generate(model, vocab, initSeq, nbIter, temperature)
   local temperature = temperature or 1.
   local h0 = torch.zeros(model.dimh)
   local currSeq = initSeq
   local inputs = initSeq
   local gentext = ''
   for inter = 1,nbIter do
      local outputs = model:forward(inputs)
      local seqSize = #outputs
      local last_out = outputs[seqSize]
      local nextDistribution = torch.exp(torch.div(last_out, temperature))
      nextDistribution:div(torch.sum(nextDistribution))
      local pred
      if true then
         pred = torch.multinomial(nextDistribution:float(), 1):resize(1):float()[1]
      else
         local max, argmax = torch.max(nextDistribution:float(), 1)
         pred = argmax[1]
      end
      gentext = gentext .. vocab[pred]
      for k=model.nbhidden+1,seqSize do
	      inputs[k] = inputs[k+1]
      end
      nextDistribution:fill(0)
      nextDistribution[pred] = 1
      inputs[#inputs] = nextDistribution
   end
   return gentext
end

function buildReverseMapping(dataset) -- TODO local global conflict
   local reverseMapping = {}
   for char, index in pairs(dataset.vocab_mapping) do
      reverseMapping[index] = char
   end
   dataset.reverseMapping = reverseMapping
end

local seqSize = 10
dataset = CharLMMinibatchLoader.create('data.t7', 'vocab.t7', 1, seqSize)
local dimx = dataset.vocab_size
local dimh = 100
buildReverseMapping(dataset)

-- model --
model = create_rnn(dimx, dimh, seqSize)
criterion = parallelNLLCriterion(seqSize)

-- train --
prepareDataset(dataset, model)


train(dataset, model, criterion, 1e-3, 2000)

-- generate --
initSeq = dataset.inputs[1]
for i = 1,10 do
   local output = generate(model, dataset.reverseMapping, initSeq, 300, .5)
   print(i..']', output)
end

torch.save('model.t7', model)
