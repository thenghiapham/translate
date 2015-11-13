require 'util.misc'

local EncoderLSTM = require 'model.EncoderLSTM'
local JointAlignDecodeLSTM = require 'model.JointAlignDecodeLSTM'
local Embedding = require 'model.Embedding'

local Encoder = require 'model.Encoder'
local Decoder = require 'model.Decoder'

local model_utils = require 'util.model_utils'

local TranslationModel = {}

