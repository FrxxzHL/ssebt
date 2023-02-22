# Text Style Transfer Back-Translation
## Introduction

Obtaining bilingual data is always harder than monolingual ones, so researchers have long exploited how to enhance machine translation performances using monolingual data. One of the most widely-used strategy is Back Translation (BT).

In terms of text style, models that use BT are usually trained on three types of data. Real parallel data constitutes the first two types: natural source with human-translated target ($Nature\rightarrow HT$) or human-translated source with natural target ($HT\rightarrow Nature$). Back translation data constitutes the third type: machine-translated source with natural target ($MT\rightarrow Nature$), as shown in the flowing figure.

![image](https://github.com/FrxxzHL/ssebt/blob/main/mt-ht-nature.PNG)

 We find that when the input style is close to $Nature$, the output is biased towards $HT$; and when the input style is closed to $HT$, the output is biased towards $Nature$ (for details, see Section \ref{sec:6.1}). Since the input used to generate BT data is $Nature$, the output is close to $HT$. So BT mainly improves the translation of translation-liked inputs. For natural inputs, BT brings only slight improvements and sometimes even adverse effects. However, in practical use, most inputs sent to NMT models are natural language written by native speakers, rather than translation-like content. 

Therefore, we propose Text Style Transfer Back Translation (TST BT), aiming to turn $MT\rightarrow Nature$ data into $Nature\rightarrow Nature$ data to enhance the translation of $Nature$ input. However, transferring translation-like text to a natural style is a zero-shot issue, because we can hardly obtain parallel data with the same meaning but different styles ($MT$ and $Nature$).

We propose two unsupervised methods. Our experiments on high-resource and low-resource language pairs demonstrate that TST BT can significantly enhance translation of $Nature$ input on basis of BT variants while brings no adverse effect on $HT$ inputs. We also find that TST BT is effective in domain adaptation, demonstrating generalizability of our method. 

![image](https://github.com/FrxxzHL/ssebt/blob/main/sse-bt.PNG)

## Experiment Results

### TST Model
We use 24M cleaned English monolingual data to train the TST model. During training, we calculate the similarity between each pair of MT and original sentence, and then filter out about 10% pairs that differ the most.
The TST model is traning fellow:

1. Translate the EN monolingual data to DE<sub>mt</sub>.
2. Translate the DE<sub>mt</sub> data to EN<sub>mt</sub>.
3. Filter 10% pairs of <EN<sub>mt</sub>, EN> which differ the most. 
4. Traning a typical transformer-based seq2seq model using pairs of <EN<sub>mt</sub>, EN><sub>filtered</sub>

For training, we use fairseq (version=1.0.0a0), and the details of training parameters can be finded in the fellowing script.
```shell
   fairseq-train data-bin \
    --fp16 \
    --ddp-backend=no_c10d \
    --arch transformer_wmt_en_de \
    --save-dir ckpts --tensorboard-logdir tf-boards \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --min-lr 1e-09 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --encoder-layers 6 --decoder-layers 6 \
    --encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --encoder-attention-heads 16 \
    --decoder-embed-dim 1024 --decoder-ffn-embed-dim 4096 --decoder-attention-heads 16 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --update-freq 8 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-interval-updates 2000 --keep-interval-updates 10 --max-epoch 100
```

Pretrained TST model and resources is comming...


## Using TST to Boost your NMT

#### EN TST

1. Data Processing

   a. Tokenize using mosesdecoder： https://github.com/moses-smt/mosesdecoder

   ```shell
   perl mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l en < bt.en > token.en
   ```

   b. Generate subwords  using BPE： https://github.com/rsennrich/subword-nmt

   ```shell
   subword-nmt apply-bpe -c wmt2014.en-de.codefile < token.en > bpe.en
   ```

2. Enhance BT data quality using TST model

   ```python
   # Load pretrained TST model
   src_lang = "rtt"  # the mt en
   target_lang = en  # the natural en
   en2de_pre_trained = TransformerModel.from_pretrained(
       model_dir_path,
       checkpoint_file=checkpoint_file,
       data_name_or_path=model_dir_path,
       source_lang=source_lang,
       target_lang=target_lang,
       device_id=0
   )
   en2de_pre_trained.cuda()
   
   # For sampling 
   en2de_pre_trained.translate(input_a_batch, beam=1, sampling=True, sampling_topk=10)
   # For Beam search
   en2de_pre_trained.translate(input_a_batch, beam=2)
   ```
