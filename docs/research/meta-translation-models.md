# Meta translation models vs Whisper

Two Meta models get mixed up in this conversation. The one this repo already names is **NLLB** (No Language Left Behind). The one Meta compares directly to Whisper is **SeamlessM4T**, especially **SeamlessM4T v2**. A newer Meta speech model, Omnilingual ASR (November 2025), transcribes only. It does not translate.

This note is for the live caption app on the church PC: Korean sermons to English, occasional English guests, faster-whisper `large-v3` on a 6 GB laptop GPU while OBS encodes. It does not propose loading either Meta model.

## Which name is which

| Model | What it is | Hears audio? | Korean |
|---|---|---|---|
| NLLB-200 | Text translation among 200 languages. Released 2022. | No | Yes, as text. FLORES-200 code `kor_Hang`. |
| SeamlessM4T v2 | One model for speech-to-text, speech-to-speech, text-to-text, text-to-speech, and speech recognition. | Yes | Korean is speech and text, in and out. |
| Omnilingual ASR | Speech recognition for 1,600+ languages. | Yes | Transcription only. Not a translator. |

`docs/architecture.md` and GitHub issue #3 defer **NLLB** and say v1 must not load a second translation model. Whisper already turns Korean speech into English text (`task=translate`) and transcribes English guests. Whisper translates only *into* English, so English-to-Korean was the direction that needed something else. That is why NLLB is paired with `en_to_ko` in the out-of-scope list.

SeamlessM4T is the model that matches "a Meta model that might be better than Whisper." It takes speech and can write the translation itself. NLLB never sees audio, so it cannot replace Whisper. It can only translate a transcript some other model already wrote.

## NLLB-200

Primary sources: the paper [No Language Left Behind](https://arxiv.org/abs/2207.04672), the [fairseq NLLB README](https://github.com/facebookresearch/fairseq/tree/nllb), the [FLORES-200 language list](https://github.com/facebookresearch/flores/tree/main/flores200), and the [600M model card](https://huggingface.co/facebook/nllb-200-distilled-600M).

- Text in, text out. The fairseq README says the models translate between any pair of the 200 FLORES languages. Korean is in that list as `kor_Hang`, English as `eng_Latn`. Both directions of text exist. There is no speech input.
- The 600M model card says it is for **single sentence** translation, trained with inputs not over 512 tokens, and "not intended to be used for document translation." A sermon is not a pile of unrelated sentences. The card also says it was tested on Wikimedia text and is "not intended to be used with domain specific texts."
- The same card says: "NLLB-200 is a research model and is not released for production deployment."
- Sizes published in the fairseq README: MoE 54.5B, dense 3.3B, dense 1.3B, distilled 1.3B, distilled 600M. The card does not give a VRAM number. A 600M-parameter model is the only one small enough to think about beside Whisper; the 3.3B dense model is already larger than Whisper large-v3.
- License on the model card is CC-BY-NC. The fairseq README says all of these checkpoints are CC-BY-NC 4.0.

Meta does not publish a Korean-sermon comparison of NLLB against Whisper's translate task. They are different jobs. Later, SeamlessM4T's paper treats "Whisper, then NLLB" as the *cascade baseline that Seamless is trying to beat*, not as a result that NLLB beats Whisper.

## SeamlessM4T and SeamlessM4T v2

Primary sources: the v1 paper [SeamlessM4T](https://arxiv.org/abs/2308.11596) (HTML body of v3), Meta's [22 August 2023 post](https://ai.meta.com/blog/seamless-m4t/), Meta's [30 November 2023 post](https://ai.meta.com/blog/seamless-communication/) for v2 and streaming, and the [v2 model card](https://huggingface.co/facebook/seamless-m4t-v2-large).

The v2 card says one model does speech-to-speech, speech-to-text, text-to-speech, text-to-text, and speech recognition.

- 101 languages for speech input, 96 for text input and output, 35 for speech output.
- Korean row on that card: code `kor`, script `Kore`, source `Sp, Tx`, target `Sp, Tx`. Spoken Korean can come out as English text. Spoken English can come out as Korean text or Korean speech.
- Public v2 checkpoint is **SeamlessM4T-Large v2, 2.3B parameters**. The card's table also lists v1 large (2.3B) and v1 medium (1.2B). There is no v2 medium in that table.
- The Hugging Face card lists the tensor type as F32. One of the two safetensors shards, `model-00002-of-00002`, is 4.24 GB. The card does not publish a VRAM figure. A 2.3B float32 checkpoint is about 9 GB of weights before activations. That does not sit beside faster-whisper large-v3 on a 6 GB GPU that is also running OBS.

### What Meta actually measured against Whisper

FLEURS is read speech, not a sermon through a mixer. The averages are pulled up by languages where Whisper is weak. Korean is called out separately below.

v1 paper, SeamlessM4T-Large against **Whisper-Large-v2** (not large-v3), mostly on FLEURS:

- Speech recognition: average word-error-rate reduction of 45% over 77 overlapping languages.
- Speech-to-text into English, human scores: "significant improvement over Whisper-Large-v2's baseline for 7 out of 24 languages." That is 7 languages, not all 24.
- Into-English speech-to-text: +1.3 BLEU versus strong cascaded models. Into-English speech-to-speech: +2.6 ASR-BLEU. On CVSS, speech-to-speech beats Whisper-Large-v2 plus YourTTS by 8.5 ASR-BLEU.
- The cascaded baselines in that paper include Whisper-Medium plus NLLB-600M, and Whisper-Large-v2 plus NLLB-1.3B. Seamless is claimed to beat those pipelines on the reported averages. Text-to-text matches NLLB-3.3B when translating into English.
- More robust than Whisper-Large-v2 to background noise and speaker variation on their FLEURS-based tests (average improvement 38% and 49% in the paper's summary).

v2 paper ([arXiv:2312.05187](https://arxiv.org/abs/2312.05187)), averages on FLEURS:

- Speech-to-text into English, 81 languages: SeamlessM4T v2 **26.6 BLEU**. Direct Whisper-Large-v2 is 17.9. Direct Whisper-Large-v3, which they re-ran, is **16.9**, worse than v2 on this average. The strongest cascade, Whisper-Large-v2 speech recognition plus NLLB-3.3B, is 22.7. The paper says v2 is more than 17% above that cascade and more than 35% above the direct Whisper numbers.
- Speech recognition, 77 languages overlapping Whisper-Large-v2: Whisper 41.7 WER, Seamless v2 18.5. On the 60 languages from the Whisper-Large-v3 release notes, Whisper-Large-v3 is 17.2 and Seamless v2 is 12.8 (4.4 points, which is the "more than 25%" in the 30 November 2023 blog, as a relative drop). The 45% story is the wide average, not the languages Whisper already handles well.

Korean, same paper, FLEURS speech-to-text BLEU. Table 65 (Korean speech to English text), reading the two header rows as Whisper-Large direct, AudioPaLM-8B, then the NLLB cascades, then Seamless medium / large v1 / large v2:

| System | BLEU |
|---|---|
| Whisper-Large direct (column headed WL) | 21.3 |
| Whisper-Large-v2 + NLLB-3.3B | 27.41 |
| SeamlessM4T Large v2 | 24.11 |
| SeamlessM4T Large v1 | 19.17 |
| SeamlessM4T Medium | 17.26 |

On this Korean row the cascade beats Seamless, and Seamless only modestly beats direct Whisper. The paper's main table uses Whisper-Large-v2 as that direct baseline and does not publish a Korean cell for large-v3.

Table 67 (English speech to Korean text) has no direct Whisper column, because Whisper does not translate out of English. SeamlessM4T Large v2 is 12.82 BLEU. The best cascade on that row is Whisper-Medium plus NLLB-3.3B at 10.55. Every system is low. BLEU is a poor score for Korean text, and the paper is still read speech.

## Quantized checkpoints

Meta did not publish an int8, int4, or GGUF SeamlessM4T. The Hugging Face card for v2 large is float32. What Meta did publish is a runtime dtype: the [seamless_communication predict README](https://github.com/facebookresearch/seamless_communication/blob/90e2b57a/src/seamless_communication/cli/m4t/predict/README.md) constructs `Translator(..., torch.device("cuda:0"), torch.float16)`, and the v2 demo space does the same on CUDA. That is half-precision execution of the full checkpoint, not a smaller released file.

An automated Hugging Face memory bot, on the v1 large Transformers port `facebook/hf-seamless-m4t-large` (also 2.3B, not the v2 checkpoint), estimates weight size only: float32 11.78 GB, float16 5.89 GB, int8 2.94 GB, int4 1.47 GB ([discussion #19](https://huggingface.co/facebook/hf-seamless-m4t-large/discussions/19)). Those int8 and int4 rows are calculator output. No corresponding checkpoint is in Meta's model table. The bot also says to add up to about 20% for inference, and that figure is still weights, not the speech activations. Float16 at 5.89 GB already fills a 6 GB card before OBS. [CTranslate2's supported-model list](https://opennmt.net/CTranslate2/guides/transformers.html) includes NLLB and Whisper and does not include SeamlessM4T, so the int8 path used for Whisper does not apply. GGUF, AWQ, and GPTQ targets are decoder-only language models. Community "lite" Seamless repos delete the speech encoder and stay float32. They cannot hear the sermon.

NLLB does quantize. It is a normal text encoder-decoder, and CTranslate2 documents conversion of `facebook/nllb-200-distilled-600M` with a `quantization` flag of `int8`, `int8_float16`, or `float16` ([NLLB section](https://opennmt.net/CTranslate2/guides/transformers.html#nllb), [quantization page](https://opennmt.net/CTranslate2/quantization.html)). The official 600M repo on the Hub is 2.48 GB. The same style of memory bot estimates that checkpoint at 2.13 GB in float16, 1.06 GB in int8, and 544 MB in int4 ([discussion #21](https://huggingface.co/facebook/nllb-200-distilled-600M/discussions/21)). A community CTranslate2 int8 upload says the file is about 600 MB ([AlaminI/nllb-200-600M-ct2-int8](https://huggingface.co/AlaminI/nllb-200-600M-ct2-int8)). An ONNX int8 export lists about 1.14 GB ([venddair/nllb-200-distilled-600M-onnx](https://huggingface.co/venddair/nllb-200-distilled-600M-onnx)). Those two sizes are the uploaders' figures. CTranslate2's own page says this kind of quantization has little to no accuracy loss in general. Neither Meta nor that page measured Korean sermon captions.

A 600M int8 model is small enough for system RAM beside Whisper on the GPU. It is not the model in the Korean FLEURS row. That row used NLLB-3.3B. Quantizing the 600M checkpoint does not turn it into the 3.3B one. The 3.3B weights at int8 are on the order of 3 GB, a RAM candidate, not a second resident model on this 6 GB GPU next to Whisper large-v3 and OBS. The license and the "not for production, not a document model" card text still apply to the quantized copies.

## Streaming

SeamlessStreaming is a separate model in the same 30 November 2023 release. Meta says it "delivers state-of-the-art results with around two seconds of latency" and writes the translation while the person is still talking, instead of waiting for the end of the sentence. It supports speech recognition and speech-to-text for nearly 100 input and output languages. The post says it is built on SeamlessM4T v2 and fine-tuned from the offline model with their EMMA policy. It is not the same checkpoint as the offline 2.3B quality model, and the post does not give a 6 GB latency number.

Metagrafo today waits for a pause (default 500 ms) and then runs Whisper on the finished chunk. Streaming would be a different pipeline, not a drop-in weight swap.

## License

- NLLB checkpoints: CC-BY-NC 4.0 (fairseq README). The 600M card additionally says the model is for research and is not released for production.
- SeamlessM4T v2 large: `cc-by-nc-4.0` on the model card. The v1 large card states the same grant for the Seamless Communication code and weights.

CC-BY-NC 4.0 is a non-commercial grant. This note does not decide whether a church YouTube stream is commercial. The NLLB card itself rules production deployment out of the intended use. Read the license before these weights ever leave a research notebook.

## Omnilingual ASR is a different model

Meta's [10 November 2025 post](https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/) introduces Omnilingual ASR for more than 1,600 languages, including a 7B model. The post describes speech recognition, not translation. It is not the model deferred in `docs/architecture.md`, and it does not do Korean-to-English captions by itself.

## For this app

Neither model replaces faster-whisper large-v3 for live Korean-to-English captions on this PC.

- NLLB is the name already in the architecture. It is the deferred text translator for a direction Whisper cannot do (into Korean), or a second sentence-level pass after a transcript exists. Its own card says single sentences, not documents, not domain-specific text, not production. The small 600M checkpoint is the only size that could even be discussed next to Whisper, and v1 already forbids that second model.
- SeamlessM4T v2 is the speech model Meta scores against Whisper. It does support Korean speech to English text. The published checkpoint is a 2.3B float32 model, which does not fit beside Whisper on a 6 GB GPU with OBS. On Meta's own Korean FLEURS row it beats direct Whisper (24.11 vs 21.3 BLEU) and loses to Whisper plus NLLB-3.3B (27.41). That NLLB is the 3.3B checkpoint, not the 600M one, and FLEURS is not a sermon. The big average gains are on languages where Whisper is weak.
- Keeping Whisper, and translating its Korean transcript with a context-aware text model, is a different design. NLLB is a poor fit for that design because it is sentence-level and the card says it is not for document translation.

## Sources

- NLLB paper: https://arxiv.org/abs/2207.04672
- NLLB code and model table: https://github.com/facebookresearch/fairseq/tree/nllb
- FLORES-200 language list (`kor_Hang`): https://github.com/facebookresearch/flores/tree/main/flores200
- NLLB-200 distilled 600M model card: https://huggingface.co/facebook/nllb-200-distilled-600M
- SeamlessM4T v1 paper: https://arxiv.org/abs/2308.11596
- SeamlessM4T v1 blog (22 August 2023): https://ai.meta.com/blog/seamless-m4t/
- Seamless v2 paper, including Tables 65 and 67: https://arxiv.org/abs/2312.05187 and https://arxiv.org/html/2312.05187v1
- Seamless blog (30 November 2023), streaming and the Whisper-Large-v3 claim: https://ai.meta.com/blog/seamless-communication/
- SeamlessM4T v2 model card (Korean row, 2.3B, CC-BY-NC 4.0): https://huggingface.co/facebook/seamless-m4t-v2-large
- Omnilingual ASR blog (10 November 2025): https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/
- Seamless float16 runtime: https://github.com/facebookresearch/seamless_communication/blob/90e2b57a/src/seamless_communication/cli/m4t/predict/README.md
- Seamless v1 large memory estimate (not a released int8 file): https://huggingface.co/facebook/hf-seamless-m4t-large/discussions/19
- CTranslate2 supported models, including NLLB and not Seamless: https://opennmt.net/CTranslate2/guides/transformers.html
- CTranslate2 quantization types: https://opennmt.net/CTranslate2/quantization.html
- NLLB 600M memory estimate: https://huggingface.co/facebook/nllb-200-distilled-600M/discussions/21
- Community NLLB 600M CTranslate2 int8: https://huggingface.co/AlaminI/nllb-200-600M-ct2-int8
- Community NLLB 600M ONNX int8: https://huggingface.co/venddair/nllb-200-distilled-600M-onnx
