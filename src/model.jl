struct AttentionPool{P,M}
  pos::P
  mhead::M
end

function AttentionPool(spacial_dim, num_head, embed_dim, output_dim)
  # i_dim = 
  positional_encoding = randn(embed_dim, round(spacial_dim ^ 2 + 1, RoundDown)) ./ sqrt.(embed_dim)
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

struct VisionTransformer{Co,C,PO,LN1,T,LN2,P}
  conv1::Co
  class_embedding::C
  positional_embedding::PO
  ln_pre::LN1
  transformer::T
  ln_post::LN2
  proj::P
end

function VisionTransformer(input_res, patch_size,
                           width, layers, heads,
                           output_dim)
  k = patch_size isa Int ? (patch_size, patch_size) : patch_size
  conv1 = Conv(k, 3 => width, stride = k, bias = false)
  scale = 1 / sqrt(width)
  class_embedding = randn(width) .* scale
  positional_embedding = randn(width, Int(round(input_res / patch_size, RoundDown) ^ 2 + 1)) .* scale
  ln_pre = LayerNorm(width)
  transformer = CLIPTransformer(width, layers, heads)
  ln_post = LayerNorm(width)
  proj = Dense(randn(output_dim, width) .* scale)
  VisionTransformer(conv1, class_embedding, positional_embedding,
                    ln_pre, transformer, ln_post, proj)
end

function (vt::VisionTransformer)(x)
  c_out = vt.conv1(x)
  re_c_out = reshape(c_out, :, size(c_out)[end-1:end]...)  
  p_c_out = permutedims(re_c_out, (2,1,3))
  cat_x = cat(reshape(repeat(vt.class_embedding, size(x)[end]), :, 1, size(x)[end]), p_c_out, dims = 2)
  cat_x2 = cat_x .+ vt.positional_embedding
  ln1_pre_out = vt.ln_pre(cat_x2)
  # transformer_out = vt.transformer(permutedims(ln1_pre_out, (2,1,3)))
  transformer_out = vt.transformer(ln1_pre_out)  

  # https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L235
  # CLIP seems to pull out only the first channel - why?
  ln_post_out = vt.ln_post(transformer_out[:, 1, :]) 
  vt.proj(ln_post_out)
end
