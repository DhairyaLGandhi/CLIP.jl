struct AttentionPool{P,M}
  pos::P
  mhead::M
end

function AttentionPool(spacial_dim, num_head, embed_dim, output_dim)
  # i_dim = 
  positional_encoding = randn(embed_dim, spacial_dim ^ 2 + 1) ./ sqrt.(embed_dim)
  mha = MultiheadAttention(num_head, embed_dim, embed_dim, isnothing(output_dim) ? embed_dim : output_dim)
  AttentionPool(positional_encoding, mha)
end

function (ap::AttentionPool)(x)
  x_ = x # permutedims(x, (1,2,4,3))
  x__ = vcat(mean(x_, dims = 3), x)
  x___ = x__ .+ ap.pos
  ap.mhead(x___[1, :, :, :], x___, x___)
end

struct ResidualAttentionBlock{A,LN1,LN2,NN,M}
  attn::A
  ln1::LN1
  ln2::LN2
  mlp::NN
  mask::M
end

function residual_attention(ra, x, mask)
  ra.attn(x,x,x, mask = mask)
end

function (ra::ResidualAttentionBlock)(x)
  a = x .+ residual_attention(ra, ra.ln1(x), ra.mask)
  a .+ ra.mlp(ra.ln2(a))
end

function ResidualAttentionBlock(width, heads, mask)
  attn = MultiheadAttention(heads, width, width, width)
  ln1 = LayerNorm(width)
  ln2 = LayerNorm(width)
  mlp = Chain(c_fc = Dense(width, width * 4),
              quickgelu = x -> x .* sigmoid(1.702f0 .* x),
              c_proj = Dense(width * 4, width))
  ResidualAttentionBlock(attn, ln1, ln2, mlp, mask)
end

struct CLIPTransformer{R}
  resblocks::R
end

function (ct::CLIPTransformer)(x)
  ct.resblocks(x)
end

function CLIPTransformer(width, layers, heads, mask = nothing)
  resblocks = Chain([ResidualAttentionBlock(width, heads, mask) for _ = 1:layers])
  CLIPTransformer(resblocks)
end
