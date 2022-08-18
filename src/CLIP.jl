module CLIP

using Flux, Flux.Zygote
using Transformers
using Transformers.Basic
using Transformers.Basic: MultiheadAttention
using Metalhead

using Statistics, LinearAlgebra

include("model.jl")

end # module
