local dir = "/home/nghia/Downloads/fr-en/"
local en_tok = dir .. "europarl-v7.fr-en.tok.en"
local fr_tok = dir .. "europarl-v7.fr-en.tok.fr"
local en_dict_file = dir .. "dict.10000.en"
local fr_dict_file = dir .. "dict.10000.fr"
local en_mapped_file = dir .. "europarl-v7.fr-en.idx.en"
local fr_mapped_file = dir .. "europarl-v7.fr-en.idx.fr"
local en_shuffled_file = dir .. "europarl-v7.fr-en.shuf.en"
local fr_shuffled_file = dir .. "europarl-v7.fr-en.shuf.fr"

--local Vocab = require 'util.preprocessing.vocab'
--local en_dict = Vocab.create_dictionary(en_tok)
--local top_en_dict = Vocab.top(en_dict,10000)
--Vocab.save(top_en_dict, dir .. "dict.10000.en")
--
--local fr_dict = Vocab.create_dictionary(fr_tok)
--local top_fr_dict = Vocab.top(fr_dict,10000)
--Vocab.save(top_fr_dict, dir .. "dict.10000.fr")

local DictMapper = require 'util.preprocessing.map_string_to_index'
DictMapper.map(en_tok, en_dict_file,en_mapped_file)
DictMapper.map(fr_tok, fr_dict_file,fr_mapped_file)

local Shuffler = require 'util.preprocessing.corpus_shuffler'
Shuffler.shuffle(en_mapped_file, fr_mapped_file, en_shuffled_file, fr_shuffled_file)