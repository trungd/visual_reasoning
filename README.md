# Visual Reasoning

### Supported Models
- [x] MAC
- [ ] Neural State Machine
- [ ] Bilinear Attention Model

### Supported Data Sets
- [x] CLEVR
- [x] GQA
  
### Train model

```
dlex train model_configs/tf_mac_clevr.yml
dlex train model_configs/tf_mac_gqa.yml
dlex train model_configs/torch_ban_gqa.yml --env lstm
```