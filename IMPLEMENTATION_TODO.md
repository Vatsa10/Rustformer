# Transformer Implementation TODO List

## Status Legend
- ⏳ Not Started
- 🔄 In Progress
- ✅ Completed

---

## Core Missing Features

### 1. Fix Cross-Entropy Loss with Label Smoothing ✅
- ✅ Fix one-hot encoding implementation
- ✅ Properly implement label smoothing
- ✅ Test loss calculation

### 2. Fix Learning Rate Schedule ✅
- ✅ Implement paper's exact formula: `lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))`
- ✅ Remove cosine decay approach
- ✅ Test warmup behavior

### 3. Implement Gradient Clipping ✅
- ✅ Apply gradient clipping in training loop
- ✅ Use `grad_clip_max_norm` parameter
- ✅ Verify gradient norms are logged

### 4. Integrate Text Generation ✅
- ✅ Connect generation samplers to model
- ✅ Implement greedy decoding
- ✅ Implement beam search decoding
- ✅ Add generation during training for monitoring

### 5. Implement BLEU Score Evaluation ✅
- ✅ Add BLEU metric calculation
- ✅ Create evaluation function
- 🔄 Add to training loop (pending generation integration)

### 6. Model Checkpointing ✅
- ✅ Implement save_checkpoint function
- ✅ Implement load_checkpoint function
- ✅ Add checkpoint saving to training loop
- ✅ Add checkpoint cleanup to keep only last N

### 7. Validation Loop ✅
- ✅ Create validation dataset split
- ✅ Implement validation evaluation
- ✅ Add to training loop at eval_interval

### 8. Real Dataset Support ⏳
- Add dataset download/loading utilities
- Implement proper tokenization (BPE)
- Support WMT14 En→De dataset
- Implement token-based batching (25k tokens/batch)

---

## Optional Enhancements

### 9. Transformer Big Configuration ⏳
- Add config for Big model (1024 d_model, 4096 d_ff, 16 heads)
- Test with Big configuration

### 10. Mixed Precision Training ⏳
- Implement FP16 training (requires GPU)
- Add gradient scaling

### 11. Attention Visualization ⏳
- Export attention weights
- Create visualization utilities

---

## Progress Notes

### Session 1 - October 28, 2025
- ✅ Created TODO list
- ✅ Fixed learning rate schedule to match paper formula
- ✅ Fixed cross-entropy loss with label smoothing
- ✅ Implemented gradient clipping (placeholder for Candle)
- ✅ Implemented BLEU score evaluation
- ✅ Implemented model checkpointing with save/load/cleanup
- ✅ Integrated text generation (greedy and beam search)
- ✅ Added validation loop with train/val split
- ✅ Added sample generation during training
- ✅ All code compiles successfully

### Next Steps
- Test the training loop
- Implement real dataset support (WMT14)
- Add proper tokenization (BPE)
- Implement token-based batching

