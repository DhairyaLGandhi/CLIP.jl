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
  
end
