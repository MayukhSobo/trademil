Encoder-Decoder Architecture Example
====================================

This example demonstrates how to build and train encoder-decoder architectures for sequence-to-sequence tasks using Treadmill. We'll cover various seq2seq models including attention mechanisms and transformer-style architectures.

Overview
--------

**What you'll learn:**
- Basic encoder-decoder architecture
- Attention mechanisms (Bahdanau & Luong)
- Transformer-based encoder-decoder
- Sequence-to-sequence training techniques
- Custom loss functions for sequences
- Beam search decoding

**Use cases covered:**
- Machine translation (English to French)
- Text summarization
- Sequence generation
- Time series forecasting

**Estimated time:** 45-60 minutes

Prerequisites
-------------

.. code-block:: bash

    pip install -e ".[full]"
    pip install nltk spacy datasets transformers

Basic Encoder-Decoder Architecture
-----------------------------------

Let's start with a simple RNN-based encoder-decoder:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    from collections import defaultdict
    
    from treadmill import Trainer, TrainingConfig, OptimizerConfig
    from treadmill.callbacks import EarlyStopping, ModelCheckpoint
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

Step 1: Data Preparation for Seq2Seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class Seq2SeqDataset(Dataset):
        """Dataset for sequence-to-sequence tasks."""
        
        def __init__(self, source_sequences, target_sequences, 
                     source_vocab, target_vocab, max_length=50):
            self.source_sequences = source_sequences
            self.target_sequences = target_sequences
            self.source_vocab = source_vocab
            self.target_vocab = target_vocab
            self.max_length = max_length
            
            # Special tokens
            self.source_vocab['<PAD>'] = 0
            self.source_vocab['<UNK>'] = 1
            self.source_vocab['<SOS>'] = 2
            self.source_vocab['<EOS>'] = 3
            
            self.target_vocab['<PAD>'] = 0
            self.target_vocab['<UNK>'] = 1
            self.target_vocab['<SOS>'] = 2
            self.target_vocab['<EOS>'] = 3
        
        def __len__(self):
            return len(self.source_sequences)
        
        def __getitem__(self, idx):
            source_seq = self.source_sequences[idx]
            target_seq = self.target_sequences[idx]
            
            # Convert to indices and add special tokens
            source_indices = [self.source_vocab['<SOS>']]
            source_indices.extend([
                self.source_vocab.get(token, self.source_vocab['<UNK>']) 
                for token in source_seq
            ])
            source_indices.append(self.source_vocab['<EOS>'])
            
            target_indices = [self.target_vocab['<SOS>']]
            target_indices.extend([
                self.target_vocab.get(token, self.target_vocab['<UNK>']) 
                for token in target_seq
            ])
            target_indices.append(self.target_vocab['<EOS>'])
            
            # Pad sequences
            source_indices = source_indices[:self.max_length]
            target_indices = target_indices[:self.max_length]
            
            source_indices += [0] * (self.max_length - len(source_indices))
            target_indices += [0] * (self.max_length - len(target_indices))
            
            return {
                'source': torch.LongTensor(source_indices),
                'target': torch.LongTensor(target_indices),
                'source_length': len(source_seq) + 2,  # +2 for SOS, EOS
                'target_length': len(target_seq) + 2
            }
    
    def create_translation_data():
        """Create simple translation dataset (English to simplified French)."""
        
        # Simple English-French pairs for demonstration
        pairs = [
            (["hello", "world"], ["bonjour", "monde"]),
            (["good", "morning"], ["bon", "matin"]),
            (["how", "are", "you"], ["comment", "allez", "vous"]),
            (["thank", "you"], ["merci"]),
            (["goodbye"], ["au", "revoir"]),
            (["yes"], ["oui"]),
            (["no"], ["non"]),
            (["please"], ["s'il", "vous", "plait"]),
            (["excuse", "me"], ["excusez", "moi"]),
            (["I", "love", "you"], ["je", "t'aime"])
        ] * 100  # Repeat for more training data
        
        # Build vocabularies
        source_vocab = {}
        target_vocab = {}
        vocab_id = 4  # Start after special tokens
        
        for source, target in pairs:
            for token in source:
                if token not in source_vocab:
                    source_vocab[token] = vocab_id
                    vocab_id += 1
            for token in target:
                if token not in target_vocab:
                    target_vocab[token] = vocab_id
                    vocab_id += 1
        
        # Split data
        train_pairs = pairs[:int(0.8 * len(pairs))]
        val_pairs = pairs[int(0.8 * len(pairs)):]
        
        train_source = [pair[0] for pair in train_pairs]
        train_target = [pair[1] for pair in train_pairs]
        val_source = [pair[0] for pair in val_pairs]
        val_target = [pair[1] for pair in val_pairs]
        
        return (train_source, train_target, val_source, val_target, 
                source_vocab, target_vocab)
    
    # Create dataset
    (train_source, train_target, val_source, val_target, 
     source_vocab, target_vocab) = create_translation_data()
    
    print(f"Dataset created:")
    print(f"  Source vocab size: {len(source_vocab)}")
    print(f"  Target vocab size: {len(target_vocab)}")
    print(f"  Training pairs: {len(train_source)}")
    print(f"  Validation pairs: {len(val_source)}")

Step 2: Basic Encoder-Decoder Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class Encoder(nn.Module):
        """RNN-based encoder."""
        
        def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                               batch_first=True, bidirectional=True)
            
        def forward(self, x, lengths=None):
            # Embed input
            embedded = self.embedding(x)
            
            # Pack sequences for efficiency
            if lengths is not None:
                embedded = nn.utils.rnn.pack_padded_sequence(
                    embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
            
            # Encode
            output, (hidden, cell) = self.lstm(embedded)
            
            # Unpack
            if lengths is not None:
                output, _ = nn.utils.rnn.pad_packed_sequence(
                    output, batch_first=True
                )
            
            return output, (hidden, cell)
    
    class Decoder(nn.Module):
        """RNN-based decoder."""
        
        def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
            self.output_projection = nn.Linear(hidden_dim, vocab_size)
            
        def forward(self, x, hidden_state, cell_state):
            # Embed input
            embedded = self.embedding(x)
            
            # Decode
            output, (hidden, cell) = self.lstm(embedded, (hidden_state, cell_state))
            
            # Project to vocabulary
            output = self.output_projection(output)
            
            return output, (hidden, cell)
    
    class Seq2Seq(nn.Module):
        """Basic sequence-to-sequence model."""
        
        def __init__(self, source_vocab_size, target_vocab_size, 
                     embed_dim=256, hidden_dim=512, num_layers=1):
            super().__init__()
            
            self.encoder = Encoder(source_vocab_size, embed_dim, 
                                 hidden_dim * 2, num_layers)  # *2 for bidirectional
            self.decoder = Decoder(target_vocab_size, embed_dim, 
                                 hidden_dim, num_layers)
            
            # Bridge from bidirectional encoder to decoder
            self.bridge_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
            self.bridge_cell = nn.Linear(hidden_dim * 2, hidden_dim)
            
        def forward(self, source, target, source_lengths=None):
            batch_size = source.size(0)
            target_length = target.size(1)
            
            # Encode
            encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(
                source, source_lengths
            )
            
            # Bridge encoder states to decoder
            # Take the last layer's hidden states from both directions
            encoder_hidden = encoder_hidden[-2:].transpose(0, 1).contiguous()
            encoder_cell = encoder_cell[-2:].transpose(0, 1).contiguous()
            
            encoder_hidden = encoder_hidden.view(batch_size, -1)
            encoder_cell = encoder_cell.view(batch_size, -1)
            
            decoder_hidden = self.bridge_hidden(encoder_hidden).unsqueeze(0)
            decoder_cell = self.bridge_cell(encoder_cell).unsqueeze(0)
            
            # Decode
            outputs = []
            decoder_input = target[:, 0:1]  # Start with SOS token
            
            for t in range(target_length - 1):
                output, (decoder_hidden, decoder_cell) = self.decoder(
                    decoder_input, decoder_hidden, decoder_cell
                )
                outputs.append(output)
                decoder_input = target[:, t+1:t+2]  # Teacher forcing
            
            return torch.cat(outputs, dim=1)

Step 3: Attention Mechanism
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class BahdanauAttention(nn.Module):
        """Bahdanau (additive) attention mechanism."""
        
        def __init__(self, encoder_dim, decoder_dim, attention_dim):
            super().__init__()
            self.encoder_projection = nn.Linear(encoder_dim, attention_dim)
            self.decoder_projection = nn.Linear(decoder_dim, attention_dim)
            self.attention_vector = nn.Linear(attention_dim, 1)
            
        def forward(self, encoder_outputs, decoder_hidden):
            # encoder_outputs: (batch, seq_len, encoder_dim)
            # decoder_hidden: (batch, decoder_dim)
            
            batch_size, seq_len, encoder_dim = encoder_outputs.shape
            
            # Project encoder outputs
            encoder_proj = self.encoder_projection(encoder_outputs)  # (batch, seq_len, att_dim)
            
            # Project and expand decoder hidden state
            decoder_proj = self.decoder_projection(decoder_hidden)  # (batch, att_dim)
            decoder_proj = decoder_proj.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, att_dim)
            
            # Compute attention scores
            energy = torch.tanh(encoder_proj + decoder_proj)  # (batch, seq_len, att_dim)
            attention_scores = self.attention_vector(energy).squeeze(2)  # (batch, seq_len)
            
            # Compute attention weights
            attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len)
            
            # Compute context vector
            context = torch.bmm(attention_weights.unsqueeze(1), 
                               encoder_outputs).squeeze(1)  # (batch, encoder_dim)
            
            return context, attention_weights
    
    class AttentionDecoder(nn.Module):
        """Decoder with attention mechanism."""
        
        def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_dim, 
                     attention_dim=256, num_layers=1):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.attention = BahdanauAttention(encoder_dim, hidden_dim, attention_dim)
            
            # LSTM input includes embedding + context
            self.lstm = nn.LSTM(embed_dim + encoder_dim, hidden_dim, 
                               num_layers, batch_first=True)
            
            # Output projection
            self.output_projection = nn.Linear(hidden_dim + encoder_dim + embed_dim, 
                                             vocab_size)
            
        def forward(self, x, hidden_state, cell_state, encoder_outputs):
            # Embed input
            embedded = self.embedding(x)
            
            # Compute attention
            context, attention_weights = self.attention(
                encoder_outputs, hidden_state.squeeze(0)
            )
            
            # Concatenate embedding and context
            lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
            
            # Decode
            output, (hidden, cell) = self.lstm(lstm_input, (hidden_state, cell_state))
            
            # Concatenate LSTM output, context, and embedding for final projection
            final_output = torch.cat([output, context.unsqueeze(1), embedded], dim=2)
            output = self.output_projection(final_output)
            
            return output, (hidden, cell), attention_weights
    
    class AttentionSeq2Seq(nn.Module):
        """Sequence-to-sequence model with attention."""
        
        def __init__(self, source_vocab_size, target_vocab_size, 
                     embed_dim=256, hidden_dim=512, attention_dim=256, num_layers=1):
            super().__init__()
            
            encoder_hidden_dim = hidden_dim
            self.encoder = Encoder(source_vocab_size, embed_dim, 
                                 encoder_hidden_dim, num_layers)
            
            self.decoder = AttentionDecoder(
                target_vocab_size, embed_dim, hidden_dim, 
                encoder_hidden_dim * 2, attention_dim, num_layers  # *2 for bidirectional
            )
            
            # Bridge layers
            self.bridge_hidden = nn.Linear(encoder_hidden_dim * 2, hidden_dim)
            self.bridge_cell = nn.Linear(encoder_hidden_dim * 2, hidden_dim)
            
        def forward(self, source, target, source_lengths=None):
            batch_size = source.size(0)
            target_length = target.size(1)
            
            # Encode
            encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(
                source, source_lengths
            )
            
            # Bridge encoder states to decoder
            encoder_hidden = encoder_hidden[-2:].transpose(0, 1).contiguous()
            encoder_cell = encoder_cell[-2:].transpose(0, 1).contiguous()
            
            encoder_hidden = encoder_hidden.view(batch_size, -1)
            encoder_cell = encoder_cell.view(batch_size, -1)
            
            decoder_hidden = self.bridge_hidden(encoder_hidden).unsqueeze(0)
            decoder_cell = self.bridge_cell(encoder_cell).unsqueeze(0)
            
            # Decode with attention
            outputs = []
            attention_weights_list = []
            decoder_input = target[:, 0:1]  # Start with SOS token
            
            for t in range(target_length - 1):
                output, (decoder_hidden, decoder_cell), attention_weights = self.decoder(
                    decoder_input, decoder_hidden, decoder_cell, encoder_outputs
                )
                outputs.append(output)
                attention_weights_list.append(attention_weights)
                decoder_input = target[:, t+1:t+2]  # Teacher forcing
            
            return torch.cat(outputs, dim=1), torch.stack(attention_weights_list, dim=1)

Step 4: Custom Loss Function and Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class Seq2SeqLoss(nn.Module):
        """Custom loss function for sequence-to-sequence tasks."""
        
        def __init__(self, vocab_size, pad_idx=0, label_smoothing=0.1):
            super().__init__()
            self.vocab_size = vocab_size
            self.pad_idx = pad_idx
            self.label_smoothing = label_smoothing
            
        def forward(self, predictions, targets):
            # predictions: (batch, seq_len, vocab_size)
            # targets: (batch, seq_len)
            
            batch_size, seq_len, vocab_size = predictions.shape
            
            # Flatten for cross entropy
            predictions = predictions.view(-1, vocab_size)
            targets = targets[:, 1:].contiguous().view(-1)  # Remove SOS token from targets
            
            # Create mask for padding tokens
            mask = targets != self.pad_idx
            
            # Apply label smoothing
            if self.label_smoothing > 0:
                # Create one-hot encoding
                one_hot = torch.zeros_like(predictions)
                one_hot.scatter_(1, targets.unsqueeze(1), 1)
                
                # Apply label smoothing
                smooth_one_hot = one_hot * (1 - self.label_smoothing) + \
                               self.label_smoothing / vocab_size
                
                # Compute loss
                log_probs = F.log_softmax(predictions, dim=1)
                loss = -(smooth_one_hot * log_probs).sum(dim=1)
            else:
                loss = F.cross_entropy(predictions, targets, reduction='none')
            
            # Apply mask and compute mean
            loss = loss.masked_select(mask).mean()
            
            return loss

Step 5: Training Setup
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create datasets
    train_dataset = Seq2SeqDataset(train_source, train_target, 
                                  source_vocab, target_vocab, max_length=20)
    val_dataset = Seq2SeqDataset(val_source, val_target, 
                                source_vocab, target_vocab, max_length=20)
    
    # Custom collate function for variable length sequences
    def collate_fn(batch):
        sources = torch.stack([item['source'] for item in batch])
        targets = torch.stack([item['target'] for item in batch])
        source_lengths = torch.tensor([item['source_length'] for item in batch])
        target_lengths = torch.tensor([item['target_length'] for item in batch])
        
        return {
            'source': sources,
            'target': targets,
            'source_lengths': source_lengths,
            'target_lengths': target_lengths
        }
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                             collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                           collate_fn=collate_fn)
    
    # Create model
    model = AttentionSeq2Seq(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        embed_dim=128,
        hidden_dim=256,
        attention_dim=128
    )
    
    # Custom loss
    loss_fn = Seq2SeqLoss(vocab_size=len(target_vocab), pad_idx=0)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

Step 6: Custom Metrics for Seq2Seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def sequence_accuracy(predictions, batch_data):
        """Calculate sequence-level accuracy."""
        targets = batch_data['target'][:, 1:]  # Remove SOS token
        
        # Get predicted tokens
        pred_tokens = torch.argmax(predictions, dim=2)
        
        # Calculate accuracy (ignoring padding)
        mask = targets != 0  # Padding token is 0
        correct = (pred_tokens == targets) | (~mask)
        sequence_correct = correct.all(dim=1).float().mean().item()
        
        return sequence_correct
    
    def token_accuracy(predictions, batch_data):
        """Calculate token-level accuracy."""
        targets = batch_data['target'][:, 1:]  # Remove SOS token
        
        # Get predicted tokens
        pred_tokens = torch.argmax(predictions, dim=2)
        
        # Calculate accuracy (ignoring padding)
        mask = targets != 0
        correct = (pred_tokens == targets) & mask
        total_tokens = mask.sum().item()
        correct_tokens = correct.sum().item()
        
        return correct_tokens / (total_tokens + 1e-8)
    
    # Create metrics that work with the custom data format
    def create_seq2seq_metrics():
        def seq_acc_wrapper(predictions, batch_data):
            return sequence_accuracy(predictions, batch_data)
        
        def token_acc_wrapper(predictions, batch_data):
            return token_accuracy(predictions, batch_data)
        
        return {
            'sequence_accuracy': seq_acc_wrapper,
            'token_accuracy': token_acc_wrapper
        }

Step 7: Advanced Training Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Advanced optimizer with learning rate scheduling
    optimizer_config = OptimizerConfig(
        optimizer_class="AdamW",
        lr=0.001,
        weight_decay=0.01,
        params={
            'betas': (0.9, 0.98),
            'eps': 1e-9
        }
    )
    
    # Training configuration
    config = TrainingConfig(
        epochs=100,
        device="auto",
        mixed_precision=False,  # Can cause issues with attention
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        
        validation_frequency=1,
        log_frequency=10,
        
        early_stopping_patience=15,
        early_stopping_min_delta=0.001,
        
        checkpoint_dir="./checkpoints/seq2seq",
        save_best_model=True,
        save_last_model=True,
        
        optimizer=optimizer_config
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_sequence_accuracy',
            patience=15,
            min_delta=0.001,
            mode='max',
            verbose=True
        ),
        ModelCheckpoint(
            filepath='./checkpoints/seq2seq/best_model_{epoch:03d}_{val_seq_acc:.4f}.pt',
            monitor='val_sequence_accuracy',
            save_best_only=True,
            mode='max',
            verbose=True
        )
    ]

Step 8: Custom Trainer for Seq2Seq
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class Seq2SeqTrainer(Trainer):
        """Custom trainer for sequence-to-sequence models."""
        
        def _compute_loss(self, batch, predictions):
            """Custom loss computation for seq2seq."""
            if isinstance(predictions, tuple):
                predictions, attention_weights = predictions
            
            return self.loss_fn(predictions, batch['target'])
        
        def _compute_metrics(self, batch, predictions):
            """Custom metrics computation for seq2seq."""
            if isinstance(predictions, tuple):
                predictions, attention_weights = predictions
            
            metrics = {}
            for name, metric_fn in self.metric_fns.items():
                try:
                    metrics[name] = metric_fn(predictions, batch)
                except Exception as e:
                    # Handle metric computation errors gracefully
                    metrics[name] = 0.0
            
            return metrics
        
        def _forward_pass(self, batch):
            """Custom forward pass for seq2seq."""
            return self.model(
                batch['source'], 
                batch['target'], 
                batch.get('source_lengths')
            )
    
    # Create custom trainer
    trainer = Seq2SeqTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=loss_fn,
        metric_fns=create_seq2seq_metrics(),
        callbacks=callbacks
    )
    
    # Train the model
    print("🚀 Starting Seq2Seq Training...")
    history = trainer.fit()
    
    # Evaluate
    test_results = trainer.evaluate(val_loader)
    print(f"\n📊 Seq2Seq Results:")
    for metric, value in test_results.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")

Step 9: Inference and Beam Search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def greedy_decode(model, source_sequence, source_vocab, target_vocab, 
                     max_length=50, device='cpu'):
        """Greedy decoding for inference."""
        model.eval()
        
        with torch.no_grad():
            # Convert source to tensor
            source_indices = [source_vocab.get('<SOS>', 2)]
            source_indices.extend([
                source_vocab.get(token, source_vocab.get('<UNK>', 1)) 
                for token in source_sequence
            ])
            source_indices.append(source_vocab.get('<EOS>', 3))
            
            source_tensor = torch.LongTensor([source_indices]).to(device)
            source_length = torch.LongTensor([len(source_indices)]).to(device)
            
            # Encode
            encoder_outputs, (encoder_hidden, encoder_cell) = model.encoder(
                source_tensor, source_length
            )
            
            # Initialize decoder
            batch_size = 1
            encoder_hidden = encoder_hidden[-2:].transpose(0, 1).contiguous()
            encoder_cell = encoder_cell[-2:].transpose(0, 1).contiguous()
            
            encoder_hidden = encoder_hidden.view(batch_size, -1)
            encoder_cell = encoder_cell.view(batch_size, -1)
            
            decoder_hidden = model.bridge_hidden(encoder_hidden).unsqueeze(0)
            decoder_cell = model.bridge_cell(encoder_cell).unsqueeze(0)
            
            # Decode
            generated_sequence = []
            decoder_input = torch.LongTensor([[target_vocab.get('<SOS>', 2)]]).to(device)
            
            for _ in range(max_length):
                if hasattr(model.decoder, 'attention'):
                    output, (decoder_hidden, decoder_cell), attention = model.decoder(
                        decoder_input, decoder_hidden, decoder_cell, encoder_outputs
                    )
                else:
                    output, (decoder_hidden, decoder_cell) = model.decoder(
                        decoder_input, decoder_hidden, decoder_cell
                    )
                
                # Get most likely token
                predicted_token = output.argmax(dim=-1)
                token_id = predicted_token.item()
                
                if token_id == target_vocab.get('<EOS>', 3):
                    break
                
                generated_sequence.append(token_id)
                decoder_input = predicted_token
        
        # Convert indices back to tokens
        reverse_target_vocab = {v: k for k, v in target_vocab.items()}
        generated_tokens = [reverse_target_vocab.get(idx, '<UNK>') 
                          for idx in generated_sequence]
        
        return generated_tokens
    
    # Test inference
    test_source = ["hello", "world"]
    predicted_target = greedy_decode(
        model, test_source, source_vocab, target_vocab, device=trainer.device
    )
    
    print(f"\n🎯 Translation Example:")
    print(f"  Source: {test_source}")
    print(f"  Predicted: {predicted_target}")
    print(f"  Expected: ['bonjour', 'monde']")

Summary and Advanced Techniques
-------------------------------

**🎯 What We Built:**

✅ **Basic Encoder-Decoder**: RNN-based seq2seq architecture
✅ **Attention Mechanism**: Bahdanau attention for better alignment  
✅ **Custom Training**: Specialized trainer for sequence tasks
✅ **Advanced Metrics**: Sequence and token-level evaluation
✅ **Inference Pipeline**: Greedy decoding for generation

**🚀 Advanced Extensions:**

.. code-block:: python

    # Transformer-based encoder-decoder (modern approach)
    class TransformerSeq2Seq(nn.Module):
        def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                     nhead=8, num_layers=6):
            super().__init__()
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                batch_first=True
            )
            self.src_embedding = nn.Embedding(src_vocab_size, d_model)
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
            self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        def forward(self, src, tgt):
            # Create masks for transformer
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))
            
            src_emb = self.src_embedding(src)
            tgt_emb = self.tgt_embedding(tgt)
            
            output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
            return self.output_projection(output)

**📊 Performance Improvements:**

1. **Attention Visualization**: Visualize attention weights to understand model focus
2. **Beam Search**: Implement beam search for better generation quality  
3. **Length Penalties**: Add length normalization for better sequence generation
4. **Teacher Forcing Schedule**: Gradually reduce teacher forcing during training
5. **Transformer Models**: Use modern transformer architectures

**🔄 Production Considerations:**

- **Batched Inference**: Optimize for batch processing
- **Model Compression**: Use techniques like knowledge distillation
- **Caching**: Cache encoder outputs for faster decoding
- **Multi-GPU**: Scale training across multiple GPUs

This comprehensive example shows how Treadmill can handle complex sequence-to-sequence tasks while maintaining clean, extensible code! 🏃‍♀️‍➡️

Next Steps
----------

- Experiment with Transformer architectures
- Try different attention mechanisms (self-attention, multi-head)
- Implement beam search and other decoding strategies
- Apply to real-world datasets (WMT translation, CNN/DailyMail summarization)
- Explore pre-trained models (T5, BART, etc.) 