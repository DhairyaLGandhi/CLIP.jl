struct AttentionPool{P,M}
  pos::P
  mhead::M
end

function AttentionPool(spacial_dim, num_head, embed_dim, output_dim)
  # i_dim = 
  positional_encoding = randn(embed_dim, spacial_dim ^ 2 + 1) ./ sqrt.(embed_dim)
  mha = MultiheadAttention(num_head, embed_dim, embed_dim, output_dim)
  AttentionPool(positional_encoding, mha)
end

function (ap::AttentionPool)(x)
  x_ = permutedims(x, (1,2,4,3))
  x__ = vcat(mean(x_, dims = 3), x)
  x___ = x__ .+ ap.pos
  ap.mhead(x___[1:1, :, :, :], x___, x___)
end
